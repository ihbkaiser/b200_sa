"""ParisKV's author router inside the common ShadowKV model/cache path.

This adapter deliberately owns no ParisKV retrieval math.  It imports the
author checkout at runtime and calls its SRHT/RxOmega encoder, fused collision
kernel, radix top-k, and fused 4-bit RaBitQ reranker.  The surrounding model
forward, RoPE, exact post-RoPE KV backing, prefix/recent policy, final gather,
and FlashAttention call are the same ones used by the other methods in this
repository.  It is therefore the quality-comparison path; the author's native
runtime remains the latency-comparison path.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys

import torch

from .streaming_cache import StreamingBlockCache


class StreamingParisKVAuthorCache(StreamingBlockCache):
    """Token-level ParisKV selection with common exact-KV lifecycle."""

    block_selection = False

    def __init__(
        self,
        config: object,
        *,
        author_root: str,
        collision_ratio: float | None = None,
        candidate_ratio: float | None = None,
        **kwargs,
    ) -> None:
        if not author_root or not os.path.isdir(author_root):
            raise ValueError("ParisKV author checkout not found")
        super().__init__(config, **kwargs)
        self.author_root = os.path.abspath(author_root)
        self.collision_ratio = collision_ratio
        self.candidate_ratio = candidate_ratio

        # Import, rather than copy, the authors' implementation.  Put their
        # project root first because polar_cache uses absolute cache_hub imports.
        if self.author_root not in sys.path:
            sys.path.insert(0, self.author_root)
        author = importlib.import_module("cache_hub.polar_cache")
        self._author_module = author

        codebook_path = os.path.join(
            self.author_root,
            "turboquant",
            "codebooks",
            "codebook_d128_m8_Kr1_Kw256_rabitq_sign.json",
        )
        with open(codebook_path, encoding="utf-8") as stream:
            codebook_config = json.load(stream)["config"]
        if int(codebook_config["d"]) != self.head_dim:
            raise ValueError(
                f"ParisKV codebook d={codebook_config['d']} does not match "
                f"head_dim={self.head_dim}"
            )

        # Construct a router-only instance of the authors' class.  Calling its
        # full constructor would allocate a second exact KV cache and reintroduce
        # the native cache lifecycle that this adapter is meant to exclude.
        core = object.__new__(author.polar_cache)
        core.device = self.compute_device
        core.dtype = self.dtype
        core.batch_size = self.batch_size
        core.kv_head = self.num_key_value_heads
        core.num_heads = self.num_attention_heads
        core.group_size = self.num_key_value_groups
        core.head_dim = self.head_dim
        core.layer_num = self.num_layers
        core.final_topk = self.sparse_budget
        core.polar_m = int(codebook_config["m"])
        core.polar_K_r = int(codebook_config["K_r"])
        core.polar_K_omega = int(codebook_config["K_omega"])
        core.polar_B = self.head_dim // core.polar_m
        core.quantizer = author.MultiDimBlockQuantizer(
            d=self.head_dim,
            m=core.polar_m,
            K_r=core.polar_K_r,
            K_omega=core.polar_K_omega,
            seed=42,
            codebook_path=codebook_path,
            device=self.compute_device,
        )
        core._V_omega_gpu_T = core.quantizer.V_omega_gpu.T.contiguous()
        core._hadamard_scale = 1.0 / math.sqrt(self.head_dim)
        core._mag_thresholds = torch.tensor(
            [0.0843, 0.1698, 0.2578, 0.3499, 0.4487, 0.5585, 0.6901],
            device=self.compute_device,
            dtype=torch.bfloat16,
        ).contiguous()
        core._mag_centers = torch.tensor(
            [0.0420, 0.1266, 0.2130, 0.3025, 0.3973, 0.5001, 0.6169, 0.7633],
            device=self.compute_device,
            dtype=torch.bfloat16,
        ).contiguous()
        core._bitpack_shifts = 1 << torch.arange(
            core.polar_m, device=self.compute_device, dtype=torch.uint8
        )

        from cache_hub.topk import load_radix_topk_ext
        from cache_hub.collision_fused.collison_interface import (
            load_kernel_module as load_collision_module,
            update_cache_cnt_cuda_interface,
        )
        from cache_hub.rerank.rerank import load_kernel_module as load_rerank

        core._radix_topk_ext = load_radix_topk_ext()
        core._update_cache_cnt_fused = update_cache_cnt_cuda_interface
        load_collision_module("collision.cu", "update_cache_cnt")
        core._rerank_kernel = load_rerank("rerank.cu", "rerank")
        core.cluster_key_counts_gpu = [None] * self.num_layers
        self._core = core

        prefix = (
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            core.polar_B,
            self.max_length,
        )
        self.codebook = torch.empty(
            prefix, device=self.compute_device, dtype=torch.uint8
        )
        self.block_weight = torch.empty(
            prefix, device=self.compute_device, dtype=torch.bfloat16
        )
        self.packed_4bit = torch.empty(
            (*prefix, core.polar_m // 2),
            device=self.compute_device,
            dtype=torch.uint8,
        )
        self._count_ranges: list[tuple[int, int] | None] = [
            None
        ] * self.num_layers

    def print_stats(self) -> None:
        super().print_stats()
        print(
            "PARISKV_AUTHOR_COMMON | author SRHT/collision/radix/RaBitQ | "
            "common model+RoPE+KV+gather+attention"
        )

    def _reset_metadata(self) -> None:
        # Only sealed, active positions are read, so clearing multi-GiB metadata
        # between prompts is unnecessary.  Range-dependent counts must reset.
        self._count_ranges[:] = [None] * self.num_layers

    def _build_blocks(
        self,
        layer_idx: int,
        block_ids: tuple[int, ...],
        block_keys: torch.Tensor | None = None,
    ) -> None:
        if not block_ids:
            return
        ids, keys = self._load_block_keys(layer_idx, block_ids, block_keys)
        token_keys = keys.flatten(2, 3).contiguous()
        codebook, weight, packed = self._core.batch_encode(
            token_keys, layer_idx=layer_idx, return_rabitq=True
        )
        offsets = torch.arange(self.block_size, device=ids.device)
        positions = (ids[:, None] * self.block_size + offsets[None]).reshape(-1)
        self.codebook[layer_idx].index_copy_(3, positions, codebook)
        self.block_weight[layer_idx].index_copy_(3, positions, weight)
        self.packed_4bit[layer_idx].index_copy_(3, positions, packed)
        self._count_ranges[layer_idx] = None

    def _score_blocks(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        first_block: int,
        last_block: int,
    ) -> torch.Tensor:
        del layer_idx, query_states, first_block, last_block
        raise RuntimeError("ParisKV author router selects tokens, not blocks")

    @torch.inference_mode()
    def get_retrieval_position_ids(
        self, layer_idx: int, query_states: torch.Tensor
    ) -> torch.Tensor:
        self.incoming_q_len = query_states.shape[-2]
        state = self.block_state[layer_idx]
        first_block, last_block = state.candidate_block_range
        start = first_block * self.block_size
        end = last_block * self.block_size
        token_count = end - start
        if token_count <= self.sparse_budget:
            ids = torch.arange(start, end, device=self.compute_device)
            return ids.view(1, 1, -1).expand(
                self.batch_size, self.num_key_value_heads, -1
            )

        codebook = self.codebook[layer_idx, ..., start:end]
        weight = self.block_weight[layer_idx, ..., start:end]
        packed = self.packed_4bit[layer_idx, ..., start:end, :]
        # The author kernel consumes exact cluster occupancies.  Recompute only
        # when the chronological candidate range changes (once per sealed block).
        active_range = (start, end)
        if self._count_ranges[layer_idx] != active_range:
            self._core.cluster_key_counts_gpu[layer_idx] = (
                self._core._count_cluster_keys(codebook, self.compute_device)
            )
            self._count_ranges[layer_idx] = active_range

        query = query_states.reshape(
            self.batch_size,
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.incoming_q_len,
            self.head_dim,
        ).mean(dim=(2, 3))
        adaptive_candidate, adaptive_collision = (
            self._core._get_adaptive_ratios(token_count)
        )
        candidate_ratio = (
            adaptive_candidate
            if self.candidate_ratio is None
            else self.candidate_ratio
        )
        candidate_ratio = min(
            1.0, max(candidate_ratio, self.sparse_budget / token_count)
        )
        collision_ratio = (
            adaptive_collision
            if self.collision_ratio is None
            else self.collision_ratio
        )
        selected = self._core.collision_based_topk_batch(
            query.unsqueeze(2),
            codebook,
            key_block_weight=weight,
            key_4bit=packed,
            collision_ratio=collision_ratio,
            candidate_ratio=candidate_ratio,
            layer_idx=layer_idx,
        )
        return selected + start
