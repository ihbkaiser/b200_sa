"""Standalone entry point for our adaptive centroid-LSE retrieval method.

The implementation reuses the tested centroid fitting and decode machinery,
but deliberately removes ShadowKV's permanently-resident outlier blocks.  The
zero-outlier invariant is enforced here rather than left to a launcher flag,
so results bearing this method name cannot silently receive ShadowKV's exact
outlier cache.
"""

from __future__ import annotations

import torch

from .centroid_router_cache import CentroidLSERouterShadowKVCache


class AdaptiveCentroidLSECache(CentroidLSERouterShadowKVCache):
    """Adaptive 1--S centroid LSE router with no outlier side channel.

    ``prefix_chunks`` optionally keeps the first blocks exactly in addition
    to the full dynamic sparse budget. These are deterministic prefix blocks,
    not ShadowKV's data-dependent outliers.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.prefix_chunks = int(kwargs.pop("prefix_chunks", 0))
        if self.prefix_chunks < 0:
            raise ValueError("prefix_chunks must be non-negative")
        requested = kwargs.pop("outlier_chunk", 0)
        if requested not in (0, None):
            raise ValueError(
                "AdaptiveCentroidLSECache forbids ShadowKV outlier blocks; "
                f"got outlier_chunk={requested}"
            )
        super().__init__(*args, outlier_chunk=0, **kwargs)
        if self.outlier_chunk != 0:
            raise AssertionError("adaptive centroid cache must have zero outliers")
        self.prefix_tokens = self.prefix_chunks * self.chunk_size
        # This router is independent of ShadowKV's low-rank key
        # reconstruction.  Keep the exact post-RoPE keys just as the
        # standalone streaming implementation does, so static/streaming
        # differ only in whether newly generated blocks enter the index.
        self.k_cache_exact = torch.zeros_like(self.v_cache_cpu)

    def print_stats(self):
        print(
            "ADAPTIVE_CENTROID_LSE | "
            f"sparse budget {self.sparse_budget} | chunk size {self.chunk_size} | "
            f"outlier chunks {self.outlier_chunk} | "
            f"exact prefix chunks {self.prefix_chunks} | "
            f"dynamic retrieval budget {self.sparse_budget} | "
            "key backing exact post-RoPE | "
            f"max centroids {self.n_centroids} | method {self.centroid_method} | "
            f"extra_fraction {self.split_fraction} | "
            f"temperatures {self.self_lse_temperatures}"
        )

    def prefill_kv_cache(self, *args, **kwargs):
        if self.outlier_chunk != 0:
            raise AssertionError("outlier invariant violated before prefill")
        layer_idx = args[1] if len(args) > 1 else kwargs["layer_idx"]
        key_states_roped = (
            args[2] if len(args) > 2 else kwargs["key_states_roped"]
        )
        result = super().prefill_kv_cache(*args, **kwargs)
        incoming = key_states_roped.shape[-2]
        self.k_cache_exact[layer_idx, :, :, :incoming].copy_(
            key_states_roped
        )
        if self.prefix_chunks > self.chunks:
            raise ValueError(
                f"prefix_chunks={self.prefix_chunks} exceeds eligible blocks={self.chunks}"
            )
        if self.prefix_tokens:
            new_v_cache = args[0] if args else kwargs["new_v_cache"]
            prefix_end = self.prefill_local + self.prefix_tokens
            self.k_cache_buffer[layer_idx][
                :, :, self.prefill_local:prefix_end
            ].copy_(key_states_roped[:, :, :self.prefix_tokens])
            self.v_cache_buffer[layer_idx][
                :, :, self.prefill_local:prefix_end
            ].copy_(new_v_cache[:, :, :self.prefix_tokens])
            # Keep the full dynamic retrieval budget after the exact prefix.
            self.sparse_start = prefix_end
            self.sparse_end = self.sparse_start + self.sparse_budget
        if self.sparse_start != self.prefill_local + self.prefix_tokens:
            raise AssertionError(
                "adaptive centroid cache has an unexpected exact-cache layout"
            )
        if self.sparse_end - self.sparse_start != self.sparse_budget:
            raise AssertionError("prefix cache must not reduce the retrieval budget")
        return result

    def get_svd(self, new_k_cache, layer_idx):
        """Do not build ShadowKV's low-rank key reconstruction."""
        del new_k_cache, layer_idx

    def get_key_cache(
        self,
        layer_idx,
        position_ids,
        rope_func=None,
        cos_sin_cache=None,
    ):
        """Gather exact post-RoPE prompt keys for the selected blocks."""
        del rope_func, cos_sin_cache
        gathered = self.k_cache_exact[layer_idx].gather(
            -2,
            position_ids.unsqueeze(-1).expand(
                -1, -1, -1, self.head_dim
            ),
        )
        self.k_cache_buffer[layer_idx][
            :, :, self.sparse_start:self.sparse_end
        ].copy_(gathered, non_blocking=True)
        gen_offset = (
            self.gen_offset
            if layer_idx == self.num_layers - 1
            else self.gen_offset + self.incoming_q_len
        )
        return self.k_cache_buffer[layer_idx][
            :, :, :self.sparse_end + gen_offset
        ]

    def clear(self):
        super().clear()
        self.k_cache_exact.zero_()

    def _select_router_chunks(self, layer_idx, chunk_attn):
        if not self.prefix_chunks:
            return super()._select_router_chunks(layer_idx, chunk_attn)
        source_blocks = self.k_landmark_idx[layer_idx]
        prefix_mask = source_blocks < self.prefix_chunks
        if not torch.all(prefix_mask.sum(-1) == self.prefix_chunks):
            raise AssertionError("router landmarks do not contain every prefix block")
        while prefix_mask.ndim < chunk_attn.ndim:
            prefix_mask = prefix_mask.unsqueeze(-2)
        masked_attn = chunk_attn.masked_fill(
            prefix_mask, float("-inf")
        )
        merged_results = torch.topk(
            masked_attn, k=self.select_sets, dim=-1
        ).indices
        selected = source_blocks.gather(dim=-1, index=merged_results)
        if torch.any(selected < self.prefix_chunks):
            raise AssertionError("dynamic retrieval duplicated an exact prefix block")
        return selected

    def _mask_router_block_logits(self, layer_idx, block_logits):
        if not self.prefix_chunks:
            return block_logits
        source_blocks = self.k_landmark_idx[layer_idx]
        prefix_mask = source_blocks < self.prefix_chunks
        while prefix_mask.ndim < block_logits.ndim:
            prefix_mask = prefix_mask.unsqueeze(-2)
        return block_logits.masked_fill(prefix_mask, float("-inf"))
