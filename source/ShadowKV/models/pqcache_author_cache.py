"""PQCache author-algorithm adapter for the common Qwen/Llama forward.

The implementation follows HugoZHL/PQCache's released ``pq_search.py``:
Euclidean product quantization, two sub-vectors, 64 codewords, approximate
per-query-head softmax scores summed inside each GQA group, and a 50/50 split
between retrieved and recent tokens.  Exact K/V live in pinned CPU memory and
only selected vectors are transferred for final attention.

PQCache's repository does not currently ship a LICENSE file, so no source is
copied from it.  The required ``kmeans-gpu`` component and the algorithm are
loaded independently; the author checkout is still required and commit-pinned
for provenance/audit.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

from .compat import head_dim_of
from .offload_gather import gather_blocks_reuse_uva, gather_kv


OFFICIAL_COMMIT = "0b74e12"


class PQCacheAuthorCache:
    def __init__(
        self,
        config: object,
        *,
        max_length: int,
        sparse_budget: int,
        batch_size: int = 1,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        author_root: str | None = None,
        sink_tokens: int | None = None,
        subvec: int | None = None,
        subbits: int | None = None,
        recent_ratio: float | None = None,
        seed: int | None = None,
        offload: bool = True,
    ) -> None:
        if batch_size != 1:
            raise ValueError("PQCache author adapter currently requires batch_size=1")
        author_root = author_root or os.environ.get("PQCACHE_AUTHOR_ROOT")
        if not author_root or not (
            Path(author_root).expanduser() / "vq_method" / "retrieval_based" / "pq_search.py"
        ).is_file():
            raise RuntimeError(
                "set PQCACHE_AUTHOR_ROOT to a HugoZHL/PQCache checkout "
                f"(tested commit {OFFICIAL_COMMIT})"
            )
        self.author_root = Path(author_root).expanduser().resolve()
        self.config = config
        self.max_length = int(max_length)
        self.sparse_budget = int(sparse_budget)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.offload = bool(offload)
        self.num_layers = int(config.num_hidden_layers)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = head_dim_of(config)
        self.matched_exact_regions = (
            os.environ.get("UPSTREAM_MATCHED_EXACT_REGIONS", "0") == "1"
        )
        if self.matched_exact_regions:
            self.sink_tokens = int(os.environ.get("QUEST_PREFIX_TOKENS", "32"))
        else:
            self.sink_tokens = int(
                os.environ.get("PQCACHE_SINK_TOKENS", "32")
                if sink_tokens is None else sink_tokens
            )
        self.subvec = int(
            os.environ.get("PQCACHE_SUBVECTORS", "2")
            if subvec is None else subvec
        )
        self.subbits = int(
            os.environ.get("PQCACHE_SUBBITS", "6")
            if subbits is None else subbits
        )
        self.codewords = 1 << self.subbits
        self.seed = int(
            os.environ.get("PQCACHE_SEED", "4321")
            if seed is None else seed
        )
        if self.head_dim % self.subvec:
            raise ValueError("head_dim must be divisible by PQ subvector count")
        self.subdim = self.head_dim // self.subvec
        if self.matched_exact_regions:
            # Fair-comparison lane: B is selected by the author PQ router;
            # prefix/recent tokens are exact and live outside B, as in ours.
            self.recent_tokens = int(
                os.environ.get("STREAMING_RECENT_TOKENS", "32")
            )
            self.retrieved_tokens = self.sparse_budget
        else:
            recent_ratio = float(
                os.environ.get("PQCACHE_RECENT_RATIO", "0.5")
                if recent_ratio is None else recent_ratio
            )
            self.recent_tokens = int(round(self.sparse_budget * recent_ratio))
            self.retrieved_tokens = self.sparse_budget - self.recent_tokens
        if min(self.recent_tokens, self.retrieved_tokens) <= 0:
            raise ValueError("PQCache needs positive recent and retrieved budgets")

        shape = (
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            self.max_length,
            self.head_dim,
        )
        # The authors keep K and V in pinned host memory and read the
        # selection back over PCIe.  That is the method's deployment story,
        # not part of its algorithm: the PQ codes and the router live on the
        # GPU either way, and the gather is exact from both sides.  Keeping
        # the store on the GPU lets a short-context campaign compare every
        # method in the same residency.
        if self.offload:
            self.k_cache = torch.empty(
                shape, dtype=dtype, device="cpu", pin_memory=True
            )
            self.v_cache = torch.empty(
                shape, dtype=dtype, device="cpu", pin_memory=True
            )
        else:
            self.k_cache = torch.empty(shape, dtype=dtype, device=self.device)
            self.v_cache = torch.empty(shape, dtype=dtype, device=self.device)
        self.centroids = torch.empty(
            self.num_layers,
            self.num_key_value_heads,
            self.subvec,
            self.codewords,
            self.subdim,
            dtype=dtype,
            device=self.device,
        )
        self.codes = torch.zeros(
            self.num_layers,
            self.num_key_value_heads,
            self.subvec,
            self.max_length,
            dtype=torch.uint8,
            device=self.device,
        )
        capacity = self.sink_tokens + self.retrieved_tokens + self.recent_tokens
        gather_shape = (
            self.batch_size,
            self.num_key_value_heads,
            capacity,
            self.head_dim,
        )
        self.k_gather = torch.empty(gather_shape, dtype=dtype, device=self.device)
        self.v_gather = torch.empty(gather_shape, dtype=dtype, device=self.device)
        # Cross-step reuse. The authors refetch the whole selection every step;
        # their own measurement shows 58% of it was already on the GPU. The
        # shared block kernel does the matching inside the launch, and a
        # retrieval unit of one token is just block_size=1, so no new kernel is
        # needed -- only somewhere to keep the previous step's rows per layer.
        self.reuse = self.offload and os.environ.get(
            "STREAMING_GATHER_REUSE", "1"
        ) == "1"
        if self.reuse:
            bank_shape = (2, self.num_layers) + gather_shape
            self.k_banks = torch.zeros(bank_shape, dtype=dtype, device=self.device)
            self.v_banks = torch.zeros_like(self.k_banks)
            self.bank_ids = torch.full(
                (2, self.num_layers, self.batch_size,
                 self.num_key_value_heads, capacity),
                -1, dtype=torch.long, device=self.device,
            )
            self.bank_slot = [0] * self.num_layers
        self.kv_offset = 0
        self.layer_lengths = [0] * self.num_layers
        # If the whole context fits in the exact active-token allowance, PQ is
        # both unnecessary and (for very short reasoning prompts) sometimes
        # undefined because there are fewer training points than codewords.
        # Keep those layers exact and fit PQ only if decoding later outgrows
        # the allowance.
        self.dense_warmup = [False] * self.num_layers

    def _fit_layer(self, layer_idx: int, keys: torch.Tensor) -> None:
        # Keep the optional author dependency lazy: importing the common model
        # package must not make unrelated baselines depend on PQCache.
        from kmeans_gpu import KMeans

        # The author implementation runs sklearn KMeans in parallel with
        # prefill.  This common adapter uses their declared kmeans-gpu package
        # with bounded iterations so both unsupported model families can share
        # one forward.  Search/scoring and all PQ dimensions remain identical.
        length = keys.shape[-2]
        start = min(self.sink_tokens, length)
        train = keys[0, :, start:, :].reshape(
            self.num_key_value_heads, length - start, self.subvec, self.subdim
        ).transpose(1, 2).contiguous()
        max_iter = int(os.environ.get("PQCACHE_KMEANS_ITERS", "10"))
        sample = int(os.environ.get("PQCACHE_KMEANS_SAMPLE", "8192"))
        sample = min(sample, train.shape[-2])
        for head in range(self.num_key_value_heads):
            for part in range(self.subvec):
                points = train[head, part].float()
                # The released multi-core compressor initializes every
                # independent worker with RANDOM_SEED=4321.
                np.random.seed(self.seed)
                model = KMeans(
                    n_clusters=self.codewords,
                    max_iter=max_iter,
                    tolerance=1e-4,
                    distance="euclidean",
                    sub_sampling=sample if sample < points.shape[0] else None,
                )
                labels, centers = model.fit_predict(points)
                self.centroids[layer_idx, head, part].copy_(centers.to(self.dtype))
                self.codes[layer_idx, head, part, start:length].copy_(
                    labels.to(torch.uint8)
                )

    def prefill_layer(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        length = int(key_states.shape[-2])
        self.k_cache[layer_idx, :, :, :length].copy_(key_states, non_blocking=True)
        self.v_cache[layer_idx, :, :, :length].copy_(value_states, non_blocking=True)
        capacity = self.sink_tokens + self.retrieved_tokens + self.recent_tokens
        self.dense_warmup[layer_idx] = length <= capacity
        if not self.dense_warmup[layer_idx]:
            self._fit_layer(layer_idx, key_states)
        self.layer_lengths[layer_idx] = length
        if layer_idx == self.num_layers - 1:
            self.kv_offset = length

    def _assign_codes(self, layer_idx: int, positions: slice, keys: torch.Tensor) -> None:
        count = keys.shape[-2]
        pieces = keys[0].reshape(
            self.num_key_value_heads, count, self.subvec, self.subdim
        ).transpose(1, 2)
        centers = self.centroids[layer_idx].float()
        for part in range(self.subvec):
            x = pieces[:, part].float()
            c = centers[:, part]
            similarity = (
                2 * torch.matmul(x, c.transpose(-1, -2))
                - x.square().sum(-1, keepdim=True)
                - c.square().sum(-1).unsqueeze(-2)
            )
            self.codes[layer_idx, :, part, positions].copy_(
                similarity.argmax(-1).to(torch.uint8)
            )

    def update_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
        start = self.layer_lengths[layer_idx]
        end = start + key_states.shape[-2]
        self.k_cache[layer_idx, :, :, start:end].copy_(key_states, non_blocking=True)
        self.v_cache[layer_idx, :, :, start:end].copy_(value_states, non_blocking=True)
        if self.dense_warmup[layer_idx]:
            capacity = self.sink_tokens + self.retrieved_tokens + self.recent_tokens
            if end <= capacity:
                self.layer_lengths[layer_idx] = end
                if layer_idx == self.num_layers - 1:
                    self.kv_offset = end
                return
            # This is a one-time transition: all exact keys are already in the
            # pinned backing store, so fit and encode them together.
            all_keys = self.k_cache[layer_idx, :, :, :end].to(
                self.device, non_blocking=False
            )
            self._fit_layer(layer_idx, all_keys)
            self.dense_warmup[layer_idx] = False
            self.layer_lengths[layer_idx] = end
            if layer_idx == self.num_layers - 1:
                self.kv_offset = end
            return
        self._assign_codes(layer_idx, slice(start, end), key_states)
        self.layer_lengths[layer_idx] = end
        if layer_idx == self.num_layers - 1:
            self.kv_offset = end

    def _selected_positions(
        self, layer_idx: int, query_states: torch.Tensor
    ) -> torch.Tensor:
        total = self.layer_lengths[layer_idx]
        if self.dense_warmup[layer_idx]:
            return torch.arange(total, device=self.device).expand(
                self.num_key_value_heads, -1
            ).unsqueeze(0).contiguous()
        recent_start = max(self.sink_tokens, total - self.recent_tokens)
        candidate_count = recent_start - self.sink_tokens
        take = min(self.retrieved_tokens, candidate_count)
        if take:
            query = query_states[:, :, -1, :].reshape(
                1, self.num_attention_heads, self.subvec, self.subdim
            )
            centers = self.centroids[layer_idx].repeat_interleave(
                self.num_key_value_groups, dim=0
            )
            table = torch.einsum("bhpd,hpkd->bhpk", query.float(), centers.float())
            codes = self.codes[
                layer_idx, :, :, self.sink_tokens:recent_start
            ].repeat_interleave(self.num_key_value_groups, dim=0).long()
            approx = torch.zeros(
                self.num_attention_heads,
                candidate_count,
                device=self.device,
                dtype=torch.float32,
            )
            for part in range(self.subvec):
                approx += table[0, :, part].gather(1, codes[:, part])
            per_query_mass = torch.softmax(
                approx / math.sqrt(self.head_dim), dim=-1
            )
            group_score = per_query_mass.reshape(
                self.num_key_value_heads,
                self.num_key_value_groups,
                candidate_count,
            ).sum(dim=1)
            dynamic = group_score.topk(take, dim=-1, sorted=False).indices
            dynamic = dynamic + self.sink_tokens
        else:
            dynamic = torch.empty(
                self.num_key_value_heads, 0, dtype=torch.long, device=self.device
            )
        sink = torch.arange(
            min(self.sink_tokens, total), device=self.device
        ).expand(self.num_key_value_heads, -1)
        recent = torch.arange(recent_start, total, device=self.device).expand(
            self.num_key_value_heads, -1
        )
        return torch.cat((dynamic, sink, recent), dim=-1).unsqueeze(0).contiguous()

    def decode_attend(
        self, layer_idx: int, query_states: torch.Tensor
    ) -> torch.Tensor:
        from flash_attn import flash_attn_with_kvcache

        positions = self._selected_positions(layer_idx, query_states)
        if self.reuse and positions.shape[-1] == self.bank_ids.shape[-1]:
            current = self.bank_slot[layer_idx]
            previous = 1 - current
            keys, values = gather_blocks_reuse_uva(
                self.k_cache[layer_idx],
                self.v_cache[layer_idx],
                self.k_banks[previous, layer_idx],
                self.v_banks[previous, layer_idx],
                self.bank_ids[previous, layer_idx],
                positions,
                self.k_banks[current, layer_idx],
                self.v_banks[current, layer_idx],
                self.bank_ids[current, layer_idx],
                block_size=1,
                exact_ranges=(),
            )
            self.bank_slot[layer_idx] = previous
        elif self.offload:
            keys, values = gather_kv(
                self.k_cache[layer_idx],
                self.v_cache[layer_idx],
                positions,
                self.k_gather,
                self.v_gather,
                backend="uva",
            )
        else:
            expanded = positions.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            keys = self.k_cache[layer_idx].gather(-2, expanded)
            values = self.v_cache[layer_idx].gather(-2, expanded)
        return flash_attn_with_kvcache(
            q=query_states.transpose(1, 2),
            k_cache=keys.transpose(1, 2),
            v_cache=values.transpose(1, 2),
            causal=True,
        )

    def get_kv_len(self) -> int:
        return self.kv_offset

    def traffic(self) -> dict[str, int | float]:
        return {
            "pqcache_sink_tokens_extra": self.sink_tokens,
            "pqcache_retrieved_tokens": self.retrieved_tokens,
            "pqcache_recent_tokens": self.recent_tokens,
            "pqcache_active_token_cap": (
                self.sink_tokens + self.retrieved_tokens + self.recent_tokens
            ),
            "pqcache_matched_exact_regions": self.matched_exact_regions,
            "pqcache_subvectors": self.subvec,
            "pqcache_bits_per_subvector": self.subbits,
            "pqcache_seed": self.seed,
        }

    def H2D(self) -> None:
        torch.cuda.current_stream(self.device).synchronize()

    def clear(self) -> None:
        self.kv_offset = 0
        self.layer_lengths[:] = [0] * self.num_layers
        self.dense_warmup[:] = [False] * self.num_layers

    def print_stats(self) -> None:
        print(
            "PQCacheAuthorCache | author algorithm "
            f"commit {OFFICIAL_COMMIT} | PQ {self.subvec}x{self.subbits} bit | "
            f"route budget {self.sparse_budget}; retrieved {self.retrieved_tokens}; "
            f"recent {self.recent_tokens}; sink {self.sink_tokens} | "
            f"kmeans-gpu {os.environ.get('PQCACHE_KMEANS_ITERS', '10')} iters | "
            f"seed {self.seed}"
        )
