"""Append-only RetroInfer correctness cache with exact post-RoPE backing.

This class preserves RetroInfer's defining update-segment lifecycle and its
three-way attention split.  It intentionally uses the auditable PyTorch
reference implementation; the production wave buffer and fused kernels remain
future integration work.  Consequently the public method name contains both
``reference`` and ``exactkv`` and must not be used for efficiency claims.
"""

from __future__ import annotations

import math

import torch

from .compat import head_dim_of
from .retroinfer_reference import (
    RetroInferIndex,
    rank_retroinfer_clusters,
    retroinfer_partition_attention,
    segmented_spherical_kmeans,
)


class StreamingRetroInferReferenceCache:
    """RetroInfer index that keeps growing across prompt and decode tokens."""

    def __init__(
        self,
        config: object,
        *,
        batch_size: int = 1,
        max_length: int = 32 * 1024,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        sparse_budget: int = 512,
        # prefix/recent/update_segment are the shared frame's sink, local
        # window and flush cadence. RetroInfer attends everything between the
        # index and the tip exactly, so a large update_segment silently buys it
        # a much larger exact region than the other methods get: at the paper's
        # 1024 it attended up to 1088 exact tokens on top of its budget.
        prefix_tokens: int = 4,
        recent_tokens: int = 64,
        update_segment: int = 1024,
        average_cluster_size: int = 16,
        estimation_ratio: float = 0.232,
        kmeans_iters: int = 10,
    ) -> None:
        if batch_size != 1:
            raise ValueError("RetroInfer reference currently requires batch_size=1")
        if sparse_budget <= 0:
            raise ValueError("sparse_budget must be positive")
        if prefix_tokens < 0 or recent_tokens < 0:
            raise ValueError("exact region sizes cannot be negative")
        if update_segment <= 0 or average_cluster_size <= 0:
            raise ValueError("segment and average cluster sizes must be positive")
        if update_segment % average_cluster_size:
            raise ValueError("update_segment must divide into whole clusters")
        if not 0 <= estimation_ratio <= 1:
            raise ValueError("estimation_ratio must lie in [0,1]")

        self.config = config
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device
        self.dtype = dtype
        self.sparse_budget = int(sparse_budget)
        self.prefix_tokens = int(prefix_tokens)
        self.recent_tokens = int(recent_tokens)
        self.update_segment = int(update_segment)
        self.average_cluster_size = int(average_cluster_size)
        self.estimation_ratio = float(estimation_ratio)
        self.kmeans_iters = int(kmeans_iters)

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.num_layers = config.num_hidden_layers
        self.head_dim = head_dim_of(config)
        self.max_clusters = math.ceil(max_length / average_cluster_size)

        cache_shape = (
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            self.max_length,
            self.head_dim,
        )
        self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        cluster_shape = (
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            self.max_clusters,
        )
        self.centroids = torch.zeros(
            (*cluster_shape, self.head_dim), device=device, dtype=dtype
        )
        self.value_sum = torch.zeros_like(self.centroids)
        self.cluster_size = torch.zeros(
            cluster_shape, device=device, dtype=torch.int32
        )
        self.assignments = torch.full(
            cache_shape[:-1], -1, device=device, dtype=torch.int32
        )
        self.total_tokens = [0 for _ in range(self.num_layers)]
        self.indexed_end = [self.prefix_tokens for _ in range(self.num_layers)]
        self.cluster_count = [0 for _ in range(self.num_layers)]
        self.kv_offset = 0
        self.last_exact_tokens = 0
        self.last_estimation_clusters = 0

    def print_stats(self) -> None:
        print(
            "RETROINFER_REFERENCE_STREAMING | exact KV backing | "
            f"nominal retrieval {self.sparse_budget} | update "
            f"{self.update_segment} | avg cluster {self.average_cluster_size} | "
            f"estimation ratio {self.estimation_ratio:g} | prefix "
            f"{self.prefix_tokens} | recent {self.recent_tokens}"
        )

    def get_kv_len(self) -> int:
        return self.kv_offset

    def H2D(self) -> None:
        pass

    def clear(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.centroids.zero_()
        self.value_sum.zero_()
        self.cluster_size.zero_()
        self.assignments.fill_(-1)
        self.total_tokens[:] = [0 for _ in range(self.num_layers)]
        self.indexed_end[:] = [self.prefix_tokens for _ in range(self.num_layers)]
        self.cluster_count[:] = [0 for _ in range(self.num_layers)]
        self.kv_offset = 0
        self.last_exact_tokens = 0
        self.last_estimation_clusters = 0

    @staticmethod
    def _training_segments(token_count: int, cluster_count: int) -> int:
        target = max(round(token_count / 8192), 1)
        return max(math.gcd(target, math.gcd(token_count, cluster_count)), 1)

    def _append_index(
        self,
        layer_idx: int,
        start: int,
        end: int,
        *,
        initial_build: bool,
    ) -> None:
        token_count = end - start
        if token_count <= 0 or token_count % self.average_cluster_size:
            raise ValueError("indexed region must contain whole average clusters")
        count = token_count // self.average_cluster_size
        base = self.cluster_count[layer_idx]
        if base + count > self.max_clusters:
            raise RuntimeError("RetroInfer cluster capacity exceeded")
        keys = self.k_cache[layer_idx, :, :, start:end].reshape(
            self.batch_size * self.num_key_value_heads,
            token_count,
            self.head_dim,
        )
        values = self.v_cache[layer_idx, :, :, start:end].reshape_as(keys)
        segments = (
            self._training_segments(token_count, count) if initial_build else 1
        )
        index = segmented_spherical_kmeans(
            keys,
            values,
            num_centroids=count,
            num_segments=segments,
            num_iters=self.kmeans_iters,
        )
        shaped_centroids = index.centroids.reshape(
            self.batch_size, self.num_key_value_heads, count, self.head_dim
        ).to(self.dtype)
        shaped_value_sum = index.value_sum.reshape_as(shaped_centroids).to(self.dtype)
        shaped_size = index.cluster_size.reshape(
            self.batch_size, self.num_key_value_heads, count
        ).to(torch.int32)
        shaped_assignments = index.assignments.reshape(
            self.batch_size, self.num_key_value_heads, token_count
        ).to(torch.int32) + base
        self.centroids[layer_idx, :, :, base : base + count].copy_(shaped_centroids)
        self.value_sum[layer_idx, :, :, base : base + count].copy_(shaped_value_sum)
        self.cluster_size[layer_idx, :, :, base : base + count].copy_(shaped_size)
        self.assignments[layer_idx, :, :, start:end].copy_(shaped_assignments)
        self.cluster_count[layer_idx] = base + count
        self.indexed_end[layer_idx] = end

    def prefill_kv_cache(
        self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> None:
        del query
        incoming = new_v_cache.shape[-2]
        if incoming > self.max_length:
            raise ValueError("prefill exceeds max_length")
        if self.total_tokens[layer_idx]:
            raise RuntimeError("prefill_kv_cache called twice without clear")
        self.k_cache[layer_idx, :, :, :incoming].copy_(key_states_roped)
        self.v_cache[layer_idx, :, :, :incoming].copy_(new_v_cache)
        self.total_tokens[layer_idx] = incoming
        start = min(self.prefix_tokens, incoming)
        eligible_end = max(start, incoming - self.recent_tokens)
        index_tokens = (
            (eligible_end - start) // self.update_segment * self.update_segment
        )
        end = start + index_tokens
        self.indexed_end[layer_idx] = start
        if index_tokens:
            self._append_index(
                layer_idx, start, end, initial_build=True
            )
        if layer_idx == self.num_layers - 1:
            self.kv_offset = incoming

    def update_kv_cache(
        self,
        new_k_cache: torch.Tensor,
        new_v_cache: torch.Tensor,
        layer_idx: int,
    ) -> None:
        incoming = new_k_cache.shape[-2]
        start = self.total_tokens[layer_idx]
        end = start + incoming
        if end > self.max_length:
            raise ValueError("RetroInfer streaming cache exceeds max_length")
        self.k_cache[layer_idx, :, :, start:end].copy_(new_k_cache)
        self.v_cache[layer_idx, :, :, start:end].copy_(new_v_cache)
        self.total_tokens[layer_idx] = end
        while (
            end - self.recent_tokens - self.indexed_end[layer_idx]
            >= self.update_segment
        ):
            segment_start = self.indexed_end[layer_idx]
            self._append_index(
                layer_idx,
                segment_start,
                segment_start + self.update_segment,
                initial_build=False,
            )
        if layer_idx == self.num_layers - 1:
            self.kv_offset = end

    def _index_for_layer(self, layer_idx: int) -> RetroInferIndex:
        count = self.cluster_count[layer_idx]
        start = min(self.prefix_tokens, self.total_tokens[layer_idx])
        end = self.indexed_end[layer_idx]
        groups = self.batch_size * self.num_key_value_heads
        return RetroInferIndex(
            centroids=self.centroids[layer_idx, :, :, :count].reshape(
                groups, count, self.head_dim
            ),
            value_sum=self.value_sum[layer_idx, :, :, :count].reshape(
                groups, count, self.head_dim
            ),
            assignments=(
                self.assignments[layer_idx, :, :, start:end].reshape(
                    groups, end - start
                ).long()
            ),
            cluster_size=self.cluster_size[layer_idx, :, :, :count].reshape(
                groups, count
            ).long(),
        )

    @torch.inference_mode()
    def decode_attend(
        self, layer_idx: int, query_states: torch.Tensor
    ) -> torch.Tensor:
        total = self.total_tokens[layer_idx]
        start = min(self.prefix_tokens, total)
        indexed_end = self.indexed_end[layer_idx]
        groups = self.batch_size * self.num_key_value_heads
        queries = query_states.view(
            self.batch_size,
            self.num_key_value_heads,
            self.num_key_value_groups,
            query_states.shape[-2],
            self.head_dim,
        )[:, :, :, -1].reshape(groups, self.num_key_value_groups, self.head_dim)

        prefix_k = self.k_cache[layer_idx, :, :, :start]
        prefix_v = self.v_cache[layer_idx, :, :, :start]
        suffix_k = self.k_cache[layer_idx, :, :, indexed_end:total]
        suffix_v = self.v_cache[layer_idx, :, :, indexed_end:total]
        steady_k = torch.cat((prefix_k, suffix_k), dim=-2).reshape(
            groups, -1, self.head_dim
        )
        steady_v = torch.cat((prefix_v, suffix_v), dim=-2).reshape_as(steady_k)
        count = self.cluster_count[layer_idx]
        if count == 0:
            logits = torch.einsum("gqd,gnd->gqn", queries.float(), steady_k.float())
            logits /= math.sqrt(self.head_dim)
            output = torch.softmax(logits, dim=-1) @ steady_v.float()
            self.last_exact_tokens = steady_k.shape[1]
            self.last_estimation_clusters = 0
        else:
            index = self._index_for_layer(layer_idx)
            indexed_k = self.k_cache[layer_idx, :, :, start:indexed_end].reshape(
                groups, indexed_end - start, self.head_dim
            )
            indexed_v = self.v_cache[layer_idx, :, :, start:indexed_end].reshape_as(
                indexed_k
            )
            nprobe = min(
                count,
                max(round(self.sparse_budget / self.average_cluster_size), 1),
            )
            estimation = min(
                round(count * self.estimation_ratio), count - nprobe
            )
            # Rank once and reuse the same partition for both attention and
            # traffic accounting.  The reference path used to sort twice.
            ranking = rank_retroinfer_clusters(queries, index)
            retrieve = ranking[:, :nprobe]
            estimate = ranking[:, nprobe : nprobe + estimation]
            output = retroinfer_partition_attention(
                queries,
                indexed_k,
                indexed_v,
                index,
                retrieve_ids=retrieve,
                estimation_ids=estimate,
                steady_keys=steady_k,
                steady_values=steady_v,
            )
            # Exact token traffic varies with cluster population; record it
            # separately from the nominal average-cluster budget.
            exact = index.cluster_size.gather(1, retrieve).sum(-1).amax().item()
            self.last_exact_tokens = int(exact + steady_k.shape[1])
            self.last_estimation_clusters = int(estimation)
        return output.reshape(
            self.batch_size, self.num_attention_heads, self.head_dim
        ).unsqueeze(1).to(query_states.dtype)
