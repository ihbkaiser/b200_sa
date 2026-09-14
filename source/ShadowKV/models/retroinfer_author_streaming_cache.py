"""RetroInfer's author index and kernels inside the common ShadowKV path.

This adapter owns no RetroInfer math.  It calls the authors' own code for every
step of their method:

* ``cache_hub/kmeans.py`` (Triton) for the segmented spherical k-means, the
  value-sum reduction and the reverse index, on the same mean-centred keys;
* ``construct_func``'s idea from ``wave_buffer_cpu.cpp`` -- the indexed region
  of the backing store is written in cluster order, so a cluster's vectors are
  contiguous and a retrieval record is a page, not a scattered token;
* ``retroinfer_kernels.batch_gemm_softmax`` (CUTLASS) for the cluster scores;
* ``retroinfer_kernels.gather_copy_vectors`` for the estimation zone's
  centroids, value sums and cluster sizes;
* ``weighted_flash_decoding`` (their flash-attention fork, installed under its
  own name so stock ``flash_attn`` is untouched) for both attention zones and
  the single softmax that merges them, through ``previous_out``/``previous_lse``
  exactly as their ``sparse_attention`` does.

The surrounding model forward, RoPE, exact post-RoPE KV backing, sink/local
frame, batch flush, offload transport and cross-step reuse are the ones every
other method in this repository uses, so a number produced here is comparable
with the rest of the table.  Their C++ wave-buffer LRU is the one component not
ported: the shared ``gather_blocks_reuse_uva`` already keeps the pages a
previous step fetched, which is what that LRU is for.

Deviations from the published configuration, all forced by this repository's
measurement protocol and none of them silent:

* **budget.**  The paper sets ``retrieval_budget`` as a *ratio of clusters*
  (0.018) and lets the retrieved token count float.  Here the budget ``B`` is a
  token count per KV head, shared by every method, so retrieval takes the
  highest-scoring *pages* until ``B`` tokens are reached.  The cluster that
  straddles the boundary is excluded from the estimation zone, since estimating
  it on top of the pages already fetched from it would count its mass twice.
* **frame.**  The paper's steady zone is ``static_pattern_start`` +
  ``static_pattern_end`` and its index grows every ``UPDATE_SEGMENT`` = 1024
  generated tokens.  Here the sink/local sizes and the flush cadence are the
  shared frame's, so the index grows every ``update_interval`` tokens instead.
  The index is extended to exactly the frame's candidate range at every flush,
  so no token is ever both clustered and attended exactly.
* **cluster count.**  The paper pins ``n_centroids`` to 8192 whatever the
  context length.  Here it defaults to ``tokens // average_cluster_size`` so
  the index costs memory in proportion to what it indexes; pass
  ``n_centroids`` explicitly to reproduce their fixed count.

RetroInfer copyright: Microsoft Corporation; official code is MIT licensed.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

import torch

from .streaming_cache import StreamingBlockCache


class StreamingRetroInferAuthorCache(StreamingBlockCache):
    """Author IVF index + estimation zone with the common exact-KV lifecycle."""

    # A cluster's tokens are contiguous in the backing store, so the retrieval
    # unit is a page of ``block_size`` tokens -- the authors' own unit, and the
    # one the shared block gather and its cross-step reuse already speak.
    block_selection = True

    # Finite stand-in for -inf: it keeps a masked cluster out of every
    # sum while leaving the arithmetic free of infinities to guard.
    LOGIT_FLOOR = -1.0e30

    def __init__(
        self,
        config: object,
        *,
        author_root: str,
        average_cluster_size: int = 16,
        n_centroids: int = 0,
        n_segment: int = 16,
        estimation_ratio: float = 0.232,
        kmeans_iters: int = 10,
        **kwargs,
    ) -> None:
        if not author_root or not os.path.isdir(author_root):
            raise ValueError("RetroInfer author checkout not found")
        super().__init__(config, **kwargs)
        # The cluster-ordered region is not a second store: a token's
        # chronological slot is dead the moment it is indexed, because the
        # exact regions are the sink and a suffix that never reaches back past
        # ``active_blocks``.  So the segment is permuted *in place*, and the
        # indexed range [prefix, active_end) is simultaneously the frame's
        # candidate range and the authors' cluster-ordered array.  Retrieval
        # therefore costs no memory over any other method's.
        if average_cluster_size <= 0:
            raise ValueError("average_cluster_size must be positive")
        if n_segment <= 0:
            raise ValueError("n_segment must be positive")
        if not 0.0 <= estimation_ratio <= 1.0:
            raise ValueError("estimation_ratio must lie in [0,1]")
        if kmeans_iters < 1:
            raise ValueError("kmeans_iters must be positive")

        if self.prefix_tokens % self.block_size:
            raise ValueError("prefix_tokens must be a block multiple")
        self.index_origin = self.prefix_tokens
        self.index_origin_block = self.prefix_tokens // self.block_size
        self.index_capacity = self.max_length - self.prefix_tokens
        self.author_root = os.path.abspath(author_root)
        self.average_cluster_size = int(average_cluster_size)
        self.requested_centroids = int(n_centroids)
        self.n_segment = int(n_segment)
        self.estimation_ratio = float(estimation_ratio)
        self.kmeans_iters = int(kmeans_iters)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Import, rather than copy, the authors' clustering kernels.  Loading
        # the file directly rather than as ``cache_hub.kmeans`` is deliberate:
        # their package __init__ pulls in the whole wave-buffer cache, which
        # needs both their compiled retroinfer_kernels and their fork of
        # flash-attention.  kmeans.py itself imports nothing but torch and
        # triton, and it is the component being borrowed.
        kmeans_path = os.path.join(self.author_root, "cache_hub", "kmeans.py")
        if not os.path.isfile(kmeans_path):
            raise ValueError(
                f"RetroInfer checkout has no cache_hub/kmeans.py: {kmeans_path}"
            )
        spec = importlib.util.spec_from_file_location(
            "retroinfer_author_kmeans", kmeans_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules.setdefault("retroinfer_author_kmeans", module)
        self._author_kmeans = module
        # segment_k_means itself is not called: _cluster_segment runs the
        # same pipeline from the same building blocks, minus the padded
        # inverted list.  Kept bound so a reader can diff the two.
        self._segment_k_means = module.segment_k_means

        # The authors' compiled kernels.  batch_gemm_softmax is their fused
        # CUTLASS cluster scorer, gather_copy_vectors their estimation-zone
        # gather, and weighted_flash_decoding their attention, which carries
        # the per-cluster weight and merges the two zones in one softmax.
        from retroinfer_kernels import batch_gemm_softmax, gather_copy_vectors
        from weighted_flash_decoding import weighted_flash_decoding

        self._batch_gemm_softmax = batch_gemm_softmax
        self._gather_copy_vectors = gather_copy_vectors
        self._weighted_flash_decoding = weighted_flash_decoding
        # Their copy kernels fix the vector length at 128 elements
        # (VECTOR_SIZE_CP in copy_kernel.cuh), so a model with another head
        # dimension would be silently misindexed rather than rejected.
        if self.head_dim != 128:
            raise ValueError(
                "RetroInfer's author kernels are compiled for head_dim 128, "
                f"this model has {self.head_dim}"
            )
        self.batch_groups = self.batch_size * self.num_key_value_heads
        self.dtype_min = torch.finfo(self.dtype).min
        self._scratch_clusters = 0
        self._scratch_ranked = 0
        self._scratch: dict[str, torch.Tensor] = {}

        # A prefill segment and every flush segment round their cluster count
        # to a multiple of lcm(8, n_segment), which is what the authors assert.
        self.centroid_multiple = math.lcm(8, self.n_segment)
        self.update_centroids = max(
            round(self.update_interval / self.average_cluster_size) // 8 * 8, 8
        )
        flushes = self.max_length // self.update_interval + 1
        clusters = (
            self.max_length // self.average_cluster_size
            + 2 * flushes
            + self.centroid_multiple
        )
        # The cluster axis is a CUTLASS GEMM dimension in batch_gemm_softmax,
        # which wants it aligned; their own assert requires n_centroids to be a
        # multiple of lcm(8, n_segment).  An unaligned width does not fail the
        # launch, it faults on a misaligned address partway through decode.
        self.max_clusters = (
            (clusters + self.centroid_multiple - 1)
            // self.centroid_multiple
            * self.centroid_multiple
        )

        groups_shape = (
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
        )
        self.centroids = torch.zeros(
            (*groups_shape, self.max_clusters, self.head_dim),
            device=self.compute_device,
            dtype=self.dtype,
        )
        self.value_sum = torch.zeros_like(self.centroids)
        # The authors' gather_copy_vectors reads cluster_size as metadata in
        # the cache dtype, so it is stored the way their cache stores it.
        # Sizes never exceed a few hundred, which bfloat16 represents exactly.
        self.cluster_size = torch.zeros(
            (*groups_shape, self.max_clusters),
            device=self.compute_device,
            dtype=self.dtype,
        )
        self.empty_cluster = torch.ones(
            (*groups_shape, self.max_clusters),
            device=self.compute_device,
            dtype=torch.bool,
        )
        # Cluster-ordered slot -> global cluster id.  This replaces the
        # chronological assignment vector: scoring now runs over the slots that
        # retrieval actually addresses.
        # int64 because it is a gather index on every decode step; storing it
        # narrower buys memory back at the cost of a conversion per layer per
        # step, and this path is launch-bound, not memory-bound.
        self.slot_cluster = torch.zeros(
            (*groups_shape, self.index_capacity),
            device=self.compute_device,
            dtype=torch.int64,
        )
        # Pages a cluster spans in the store.  Clusters are packed without
        # padding, so a page can straddle two of them and the count is not
        # ceil(size / block_size).
        self.cluster_pages = torch.zeros(
            (*groups_shape, self.max_clusters),
            device=self.compute_device,
            dtype=torch.int32,
        )
        self.cluster_count = [0] * self.num_layers
        self.index_fill = [0] * self.num_layers
        self.indexed_end = [self.prefix_tokens] * self.num_layers
        # Set while prefill_kv_cache holds the prompt's K/V on the GPU, so the
        # index is built from those tensors instead of reading the (possibly
        # pinned-CPU) backing store back.
        self._prefill_tensors: list[tuple | None] = [None] * self.num_layers
        self._pending_estimation: list[int | None] = [None] * self.num_layers
        self._estimation_ids: list[torch.Tensor | None] = [None] * self.num_layers
        self._author_queries: list[torch.Tensor | None] = [None] * self.num_layers
        self.last_estimation_clusters = 0
        self.last_retrieved_clusters = 0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def print_stats(self) -> None:
        super().print_stats()
        print(
            "RETROINFER_AUTHOR_COMMON | author segmented spherical k-means | "
            f"cluster-ordered store, page {self.block_size} | "
            f"avg cluster {self.average_cluster_size} | "
            f"n_centroids {self.requested_centroids or 'derived'} | "
            f"n_segment {self.n_segment} | estimation ratio "
            f"{self.estimation_ratio:g} | common KV+gather+attention"
        )

    def _reset_metadata(self) -> None:
        # The centroid and value-sum tensors are not cleared: only slots below
        # cluster_count are written, and the liveness mask decides what is read.
        #
        # That mask MUST be cleared.  The authors' kernels take the cluster axis
        # as a GEMM dimension, so scoring runs over the tensor's full width and
        # empty_cluster is the only thing keeping the unbuilt tail out of the
        # ranking.  Left stale, a shorter prompt inherits the previous prompt's
        # live flags, selects clusters whose slots now hold someone else's
        # tokens, and produces fluent garbage -- LongBench-v2 scored 0.039 this
        # way while every sibling method scored 0.35, and the first samples of
        # the run looked fine because nothing was stale yet.
        self.empty_cluster.fill_(True)
        self.cluster_pages.zero_()
        self.cluster_count[:] = [0] * self.num_layers
        self.index_fill[:] = [0] * self.num_layers
        self.indexed_end[:] = [self.prefix_tokens] * self.num_layers
        self._prefill_tensors[:] = [None] * self.num_layers
        self._pending_estimation[:] = [None] * self.num_layers
        self._estimation_ids[:] = [None] * self.num_layers
        self._author_queries[:] = [None] * self.num_layers
        self._scratch_clusters = 0
        self._scratch_ranked = 0

    def prefill_kv_cache(
        self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> None:
        self._prefill_tensors[layer_idx] = (key_states_roped, new_v_cache)
        try:
            super().prefill_kv_cache(
                new_v_cache, layer_idx, key_states_roped, query
            )
        finally:
            self._prefill_tensors[layer_idx] = None

    # ------------------------------------------------------------------ #
    # index construction
    # ------------------------------------------------------------------ #

    def _build_or_defer_blocks(
        self,
        layer_idx: int,
        block_ids: tuple[int, ...],
        block_keys: torch.Tensor,
    ) -> None:
        """Arm the flush without copying keys.

        The base class queues the sealed blocks' keys so a per-block router can
        fit them all at once.  This index is not built from the newly sealed
        blocks at all -- it is extended to whatever the frame has just made
        retrievable, which is an *older* stretch of the context -- so the queue
        only needs to remember that a flush is due.
        """
        if self.update_interval == self.block_size:
            self._build_blocks(layer_idx, block_ids, block_keys)
            return
        ids, _ = self._pending_block_builds[layer_idx]
        ids.extend(block_ids)

    def _flush_block_builds(self, layer_idx: int) -> None:
        ids, _ = self._pending_block_builds[layer_idx]
        if not ids:
            return
        self._pending_block_builds[layer_idx] = ([], [])
        self._extend_index(layer_idx)

    def _build_blocks(
        self,
        layer_idx: int,
        block_ids: tuple[int, ...],
        block_keys: torch.Tensor | None = None,
    ) -> None:
        del block_ids, block_keys
        self._extend_index(layer_idx)

    def _segment_keys_values(
        self, layer_idx: int, start: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [groups, tokens, dim] keys and values for [start, end)."""
        prefill = self._prefill_tensors[layer_idx]
        if prefill is not None:
            keys, values = prefill
            key_slice = keys[..., start:end, :]
            value_slice = values[..., start:end, :]
        else:
            if self.offload:
                # The appends that wrote this stretch were asynchronous copies
                # into pinned memory; make them visible before reading back.
                torch.cuda.current_stream(self.compute_device).synchronize()
            key_slice = self.k_cache[layer_idx, :, :, start:end]
            value_slice = self.v_cache[layer_idx, :, :, start:end]
        groups = self.batch_size * self.num_key_value_heads
        shape = (groups, end - start, self.head_dim)
        return (
            key_slice.reshape(shape).to(self.compute_device).contiguous(),
            value_slice.reshape(shape).to(self.compute_device).contiguous(),
        )

    def _centroid_count(self, tokens: int, segments: int, prefill: bool) -> int:
        # n_centroids in the paper's config pins the *prefill* index only; a
        # flush segment uses their UPDATE_CENTROIDS rule, which is the segment
        # length over the average cluster size rounded to a multiple of eight.
        if prefill and self.requested_centroids > 0:
            count = self.requested_centroids
        else:
            count = round(tokens / self.average_cluster_size)
        multiple = math.lcm(8, segments)
        count = min(count, tokens)
        count = count // multiple * multiple
        return max(count, multiple) if tokens >= multiple else 0

    @torch.inference_mode()
    def _extend_index(self, layer_idx: int) -> None:
        """Cluster everything the frame has just made retrievable."""
        state = self.block_state[layer_idx]
        start = self.indexed_end[layer_idx]
        end = state.active_blocks * self.block_size
        tokens = end - start
        if tokens <= 0:
            return
        prefill = self._prefill_tensors[layer_idx] is not None
        segments = self.n_segment if prefill else 1
        count = self._centroid_count(tokens, segments, prefill)
        if count <= 0:
            # Too short to cluster: leave it unindexed.  It stays inside the
            # exact suffix, which is where the frame already puts it.
            return
        base = self.cluster_count[layer_idx]
        if base + count > self.max_clusters:
            raise RuntimeError(
                f"RetroInfer index holds {self.max_clusters} clusters but "
                f"layer {layer_idx} needs {base + count}"
            )

        keys, values = self._segment_keys_values(layer_idx, start, end)
        # The authors centre the segment on its mean key before clustering and
        # add the mean back to the centroids; the assignment is unchanged by
        # the shift but the k-means normalisation is not.
        mean_key = keys.mean(dim=1, keepdim=True)
        centroids, value_sum, order, owners, cluster_size = self._cluster_segment(
            keys - mean_key, values, count, segments
        )
        centroids = centroids + mean_key

        shape = (self.batch_size, self.num_key_value_heads, count)
        self.centroids[layer_idx, :, :, base : base + count].copy_(
            centroids.reshape(*shape, self.head_dim)
        )
        self.value_sum[layer_idx, :, :, base : base + count].copy_(
            value_sum.reshape(*shape, self.head_dim)
        )
        self.cluster_size[layer_idx, :, :, base : base + count].copy_(
            cluster_size.reshape(shape).to(self.dtype)
        )
        self.empty_cluster[layer_idx, :, :, base : base + count].copy_(
            cluster_size.reshape(shape) == 0
        )

        # This is the authors' construct_func: the segment is written into the
        # store in cluster order, so a cluster's vectors are contiguous and a
        # retrieval record is a page rather than a scattered token.
        slot = self.index_fill[layer_idx]
        if slot % self.block_size:
            raise RuntimeError("cluster region lost its page alignment")
        if slot + tokens > self.index_capacity:
            raise RuntimeError(
                f"cluster region holds {self.index_capacity} tokens but layer "
                f"{layer_idx} needs {slot + tokens}"
            )
        index = order.unsqueeze(-1).expand(-1, -1, self.head_dim)
        ordered_keys = keys.gather(1, index).reshape(
            self.batch_size, self.num_key_value_heads, tokens, self.head_dim
        )
        ordered_values = values.gather(1, index).reshape_as(ordered_keys)
        first = self.index_origin + slot
        if first != start:
            raise RuntimeError(
                "cluster region drifted from the chronological range it "
                f"replaces: writing {first} for segment starting {start}"
            )
        # In place: the segment's own chronological slots are what the permuted
        # copy goes into.  Its source was read to the compute device first, so
        # nothing aliases.
        self.k_cache[layer_idx, :, :, start:end].copy_(
            ordered_keys, non_blocking=self.offload
        )
        self.v_cache[layer_idx, :, :, start:end].copy_(
            ordered_values, non_blocking=self.offload
        )
        self.slot_cluster[layer_idx, :, :, slot : slot + tokens].copy_(
            owners.reshape(self.batch_size, self.num_key_value_heads, tokens)
            .add(base)
        )
        # Pages a cluster spans.  Clusters are packed without padding, so the
        # page holding a cluster's first vector may also hold the tail of the
        # previous one; the count follows from the slot range, not the size.
        sizes = cluster_size.reshape(shape)
        starts = sizes.cumsum(-1) - sizes
        head = (first + starts) // self.block_size
        tail = (first + starts + sizes.clamp(min=1) - 1) // self.block_size
        self.cluster_pages[layer_idx, :, :, base : base + count].copy_(
            torch.where(sizes > 0, tail - head + 1, torch.zeros_like(sizes))
        )
        if self.offload:
            # The gather this step runs against the region just written.
            torch.cuda.current_stream(self.compute_device).synchronize()

        self.index_fill[layer_idx] = slot + tokens
        self.cluster_count[layer_idx] = base + count
        self.indexed_end[layer_idx] = end

    @torch.inference_mode()
    def _cluster_segment(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        count: int,
        segments: int,
    ) -> tuple[torch.Tensor, ...]:
        """The authors' segmented spherical k-means, ordered by a sort.

        This runs their ``segment_k_means`` -- the same uniform midpoint
        initialisation, the same ``_triton_k_means_train`` iterations with the
        same normalise flags, the same ``triton_index_add`` value reduction --
        and then departs from it in exactly one place: it does not call
        ``triton_reverse_index``.

        That kernel materialises the padded inverted list, a
        ``[groups, n_centroids, max_cluster_size]`` int32 tensor.  Its size is
        set by the *largest* cluster, so one degenerate cluster costs as much
        as if every cluster were that big.  It cost 19.13 GiB and killed a 128K
        cell: at that length one cluster had absorbed about 78 000 of 131 000
        tokens, and 8 x 8180 x 78000 x 4 bytes is what the kernel then asked
        for.  Their wave buffer needs that list; this store does not.  Sorting
        the assignment vector gives the same grouping in O(tokens).
        """
        kmeans = self._author_kmeans
        groups, tokens, dim = keys.shape
        if count % segments:
            raise ValueError("centroid count must divide into whole segments")
        # Their initialisation: centroids at uniform midpoints of the segment.
        picks = torch.arange(count, dtype=torch.float32, device=keys.device)
        picks = picks * (tokens / count) + tokens / count / 2
        centroids = torch.index_select(keys, 1, picks.to(torch.int64))

        per_segment = tokens // segments
        per_centroid = count // segments
        data = keys[:, : per_segment * segments].reshape(
            (-1, per_segment, dim)
        )
        centroids = centroids.reshape((-1, per_centroid, dim))
        assignments = torch.empty(
            (data.shape[0], data.shape[1]), dtype=torch.int32,
            device=data.device,
        )
        for _ in range(self.kmeans_iters - 1):
            centroids = kmeans._triton_k_means_train(
                data, centroids, max_idx=assignments,
                normalize_centroids=True, return_indices=False,
            )
        data = keys.reshape((-1, tokens, dim))
        centroids = centroids.reshape((-1, count, dim))
        centroids, assignments, _ = kmeans._triton_k_means_train(
            data, centroids, normalize_centroids=False, return_indices=True,
        )
        value_sum = kmeans.triton_index_add(
            values.reshape((-1, tokens, dim)), assignments, count
        )

        # Cluster order without the padded list: a stable sort of the
        # assignment vector lists every token exactly once, grouped by cluster
        # and in chronological order inside each cluster.
        assignments = assignments.to(torch.int64)
        order = assignments.argsort(dim=1, stable=True)
        owners = assignments.gather(1, order)
        cluster_size = torch.zeros(
            (groups, count), dtype=torch.int32, device=keys.device
        )
        cluster_size.scatter_add_(
            1, owners, torch.ones_like(owners, dtype=torch.int32)
        )
        return centroids, value_sum, order, owners, cluster_size

    def _ensure_scratch(self, count: int, ranked: int, estimation: int) -> None:
        """(Re)allocate the authors' fixed-shape working buffers.

        Their cache reallocates these whenever the index grows, because the
        kernels take the cluster count as a GEMM dimension and the buffers must
        be exactly that wide.  The index only grows at a flush, so this runs
        once per ``update_interval`` steps, not per step.
        """
        if self._scratch_clusters == count and self._scratch_ranked == ranked:
            return
        groups = self.batch_groups
        heads = self.num_key_value_groups
        device = self.compute_device
        tiles = (count + 255) // 256
        self._scratch = {
            "gemm": torch.zeros(
                (groups, heads, count), device=device, dtype=self.dtype
            ),
            "softmax": torch.zeros(
                (groups, heads, count), device=device, dtype=self.dtype
            ),
            "norm": torch.zeros(
                (groups, heads, tiles), device=device, dtype=torch.float32
            ),
            "sum": torch.zeros(
                (groups, heads, tiles), device=device, dtype=torch.float32
            ),
            "dist": torch.zeros(
                (groups, count), device=device, dtype=self.dtype
            ),
            "values": torch.zeros(
                (groups, ranked), device=device, dtype=self.dtype
            ),
            "indices": torch.zeros(
                (groups, ranked), device=device, dtype=torch.int64
            ),
            "es_centroids": torch.zeros(
                (groups, max(estimation, 1), 1, self.head_dim),
                device=device, dtype=self.dtype,
            ),
            "es_value_sum": torch.zeros(
                (groups, max(estimation, 1), 1, self.head_dim),
                device=device, dtype=self.dtype,
            ),
            "es_sizes": torch.zeros(
                (groups, 1, 1, max(estimation, 1)),
                device=device, dtype=self.dtype,
            ),
        }
        self._scratch_clusters = count
        self._scratch_ranked = ranked

    def _score_blocks(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        first_block: int,
        last_block: int,
    ) -> torch.Tensor:
        del layer_idx, query_states, first_block, last_block
        raise RuntimeError(
            "RetroInfer scores pages of the cluster-ordered region, not "
            "chronological blocks"
        )

    def _cluster_scores(
        self, layer_idx: int, queries: torch.Tensor, count: int
    ) -> torch.Tensor:
        """Run the authors' fused cluster scorer.

        ``batch_gemm_softmax`` computes ``softmax(Q C^T / sqrt(d))`` per query
        head in one CUTLASS kernel; summing over the heads that share a KV head
        is their ``torch.sum(softmax_o, dim=1)``.  Empty clusters are masked to
        the dtype minimum before ranking, as they are there.
        """
        scratch = self._scratch
        self._batch_gemm_softmax(
            queries,
            self.centroids[layer_idx].view(
                self.batch_groups, -1, self.head_dim
            ),
            scratch["gemm"],
            scratch["norm"],
            scratch["sum"],
            scratch["softmax"],
            self.batch_groups,
            self.num_key_value_groups,
            count,
            self.head_dim,
            self.scale,
            0.0,
        )
        distance = scratch["dist"]
        torch.sum(scratch["softmax"], dim=1, out=distance)
        distance.masked_fill_(
            self.empty_cluster[layer_idx].view(self.batch_groups, count),
            self.dtype_min,
        )
        return distance

    @torch.inference_mode()
    def _select_block_ids(
        self, layer_idx: int, query_states: torch.Tensor
    ) -> torch.Tensor:
        """Choose pages of the cluster-ordered region, and stage the zone split.

        Scoring is the authors' kernel: one score per cluster.  A page inherits
        the best score among the clusters whose vectors it holds, which is how
        "retrieve the top clusters" reads once the store is cluster-ordered and
        the unit of transfer is a page.  Scoring pages rather than clusters is
        what keeps the budget exact: no padding slot is ever attended, no page
        is fetched twice, and the count is exactly ``sparse_budget`` tokens.
        """
        self.incoming_q_len = query_states.shape[-2]
        self._pending_estimation[layer_idx] = None
        fill = self.index_fill[layer_idx]
        count = self.cluster_count[layer_idx]
        pages = fill // self.block_size
        if count == 0 or pages <= self.select_blocks:
            return torch.arange(
                self.index_origin_block,
                self.index_origin_block + pages,
                device=self.compute_device,
            ).view(1, 1, -1).expand(
                self.batch_size, self.num_key_value_heads, -1
            )

        estimation = min(
            round(count * self.estimation_ratio), max(count - 1, 0)
        )
        # Their kernels take the cluster axis as a GEMM dimension and index the
        # tensor by it, so it must be the tensor's own width.  Their cache
        # reallocates on every index growth; here the axis is simply the full
        # capacity, and the clusters not yet built are empty ones, which the
        # dtype-minimum mask already keeps out of every ranking.
        width = self.max_clusters
        # No cluster spans fewer than one page, so the page budget can never
        # reach past rank ``select_blocks``: ranking that prefix is enough.
        ranked = min(width, self.select_blocks + estimation + 1)
        self._ensure_scratch(width, ranked, estimation)

        # The authors' kernel wants the queries as (batch, 1, heads, dim); the
        # size-one dimension makes that the same buffer as (groups, 1, G, dim).
        queries = query_states[:, :, -1:].transpose(1, 2).contiguous()
        self._author_queries[layer_idx] = queries
        distance = self._cluster_scores(layer_idx, queries, width)

        slots = self.slot_cluster[layer_idx].view(self.batch_groups, -1)
        page_scores = distance.gather(1, slots[:, :fill]).view(
            self.batch_groups, pages, self.block_size
        ).amax(-1)
        selected = torch.topk(
            page_scores, k=self.select_blocks, dim=-1, sorted=False
        ).indices.add_(self.index_origin_block)
        selected = selected.view(
            self.batch_size, self.num_key_value_heads, self.select_blocks
        )
        if estimation <= 0:
            self.last_estimation_clusters = 0
            return selected

        scratch = self._scratch
        torch.topk(
            distance, ranked, dim=-1, largest=True, sorted=True,
            out=(scratch["values"], scratch["indices"]),
        )
        order = scratch["indices"]
        spans = self.cluster_pages[layer_idx].view(
            self.batch_groups, -1
        ).gather(1, order)
        covered = (spans.cumsum(-1) <= self.select_blocks).sum(-1)
        offsets = torch.arange(estimation, device=self.compute_device)
        wanted = (covered.unsqueeze(-1) + 1 + offsets).clamp_(max=ranked - 1)
        self._estimation_ids[layer_idx] = order.gather(1, wanted).contiguous()
        self._pending_estimation[layer_idx] = width
        self.last_estimation_clusters = estimation
        return selected

    @torch.inference_mode()
    def decode_attend_gathered(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        """The authors' two-zone attention, merged in one softmax.

        This is ``sparse_attention``'s tail: ``gather_copy_vectors`` pulls the
        estimation zone's three vectors per cluster, ``weighted_flash_decoding``
        turns them into an ``(out, lse)`` pair with each cluster weighted by its
        size, and the second call folds that into the attention over the pages
        and the exact regions through ``previous_out``/``previous_lse``.
        """
        groups = self.batch_groups
        tokens = key_states.shape[-2]
        queries = self._author_queries[layer_idx]
        if queries is None or queries.shape[0] != self.batch_size:
            queries = query_states[:, :, -1:].transpose(1, 2).contiguous()
        query_view = queries.view(groups, 1, self.num_key_value_groups,
                                  self.head_dim)
        keys = key_states.reshape(groups, tokens, 1, self.head_dim)
        values = value_states.reshape(groups, tokens, 1, self.head_dim)

        count = self._pending_estimation[layer_idx]
        self._pending_estimation[layer_idx] = None
        previous_out = previous_lse = None
        if count:
            scratch = self._scratch
            estimation = scratch["es_sizes"].shape[-1]
            self._gather_copy_vectors(
                self.centroids[layer_idx].view(groups, -1, self.head_dim),
                scratch["es_centroids"],
                self.value_sum[layer_idx].view(groups, -1, self.head_dim),
                scratch["es_value_sum"],
                self.cluster_size[layer_idx].view(groups, -1),
                scratch["es_sizes"],
                self._estimation_ids[layer_idx],
                groups,
                count,
                estimation,
                estimation,
                0,
                estimation,
            )
            previous_out, previous_lse = self._weighted_flash_decoding(
                query_view,
                scratch["es_centroids"],
                scratch["es_value_sum"],
                scratch["es_sizes"],
                previous_out=None,
                previous_lse=None,
                return_softmax_lse=True,
            )

        output = self._weighted_flash_decoding(
            query_view,
            keys,
            values,
            previous_out=previous_out,
            previous_lse=previous_lse,
            return_softmax_lse=False,
        )
        self.last_attention_tokens = tokens
        return output.reshape(
            self.batch_size, 1, self.num_attention_heads, self.head_dim
        )
