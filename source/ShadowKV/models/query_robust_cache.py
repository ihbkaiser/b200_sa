"""Streaming Query-Robust cache using the exact backing K/V store."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .query_robust import (
    build_query_robust_page_summaries,
    load_query_robust_asset,
    score_query_robust_pages_reference,
)
from .streaming_cache import StreamingBlockCache


class StreamingQueryRobustCache(StreamingBlockCache):
    """QR page router on the common streaming lifecycle.

    The cache keeps the complete post-RoPE K/V backing store inherited from
    :class:`StreamingBlockCache`.  QR metadata is built only for sealed pages
    that can become retrievable, and the inherited gather then reads exactly
    the pages selected by the QR score.  This is intentionally a PyTorch
    reference path; it exposes the algorithm without claiming a fused-kernel
    performance result.
    """

    def __init__(
        self,
        config: object,
        *,
        vertices_path: str | Path,
        model_id: str | None = None,
        model_fingerprint: str | None = None,
        rope_config: object | None = None,
        vertices_sha256: str | None = None,
        num_vertices: int = 32,
        solver_iters: int = 24,
        solver_lr: float = 0.25,
        score_alpha: float = 1.0,
        uniform_p: bool = False,
        summary_page_batch: int | None = None,
        **kwargs,
    ) -> None:
        if int(getattr(config, "num_attention_heads")) % int(
            getattr(config, "num_key_value_heads")
        ):
            raise ValueError("Qwen3 GQA query heads must be divisible by KV heads.")
        self.num_vertices = int(num_vertices)
        self.solver_iters = int(solver_iters)
        self.solver_lr = float(solver_lr)
        self.score_alpha = float(score_alpha)
        self.uniform_p = bool(uniform_p)
        self.summary_page_batch = int(
            summary_page_batch
            if summary_page_batch is not None
            else os.environ.get("QUERY_ROBUST_SUMMARY_PAGE_BATCH", "64")
        )
        if self.summary_page_batch <= 0:
            raise ValueError("summary_page_batch must be positive")
        if self.solver_iters <= 0 or self.solver_lr <= 0:
            raise ValueError("QR solver_iters and solver_lr must be positive")
        if not torch.isfinite(torch.tensor(self.score_alpha)):
            raise ValueError("score_alpha must be finite")

        super().__init__(config, **kwargs)
        if self.group_reduce != "max":
            raise ValueError(
                "Query-Robust uses the source method's shared max GQA route; "
                "set group_reduce='max'."
            )
        asset = load_query_robust_asset(
            vertices_path,
            num_layers=self.num_layers,
            global_num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_vertices=self.num_vertices,
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            expected_model_id=model_id,
            expected_model_fingerprint=model_fingerprint,
            expected_rope_config=rope_config,
            expected_sha256=vertices_sha256,
            device=self.compute_device,
        )
        if int(asset.vertices.shape[1]) != self.num_key_value_heads:
            raise ValueError(
                "QR local KV heads do not match runtime: "
                f"asset={asset.vertices.shape[1]} runtime={self.num_key_value_heads}."
            )
        self.query_robust_vertices = asset.vertices
        self.query_robust_num_valid_vertices = asset.num_valid_vertices
        self.query_robust_asset_meta = asset.meta
        self.query_robust_scale = float(self.head_dim) ** -0.5
        self.landmark_cache = torch.zeros(
            (
                self.num_layers,
                self.num_key_value_heads,
                self.max_blocks,
                self.head_dim,
            ),
            device=self.compute_device,
            dtype=torch.bfloat16,
        )
        self.bias_cache = torch.zeros(
            (self.num_layers, self.num_key_value_heads, self.max_blocks),
            device=self.compute_device,
            dtype=torch.float32,
        )
        self.epsilon_cache = torch.zeros_like(self.bias_cache)
        self.duality_gap_cache = torch.zeros_like(self.bias_cache)
        self.metadata_valid = torch.zeros(
            (self.num_layers, self.num_key_value_heads, self.max_blocks),
            device=self.compute_device,
            dtype=torch.bool,
        )

    def _reset_metadata(self) -> None:
        if hasattr(self, "metadata_valid"):
            self.landmark_cache.zero_()
            self.bias_cache.zero_()
            self.epsilon_cache.zero_()
            self.duality_gap_cache.zero_()
            self.metadata_valid.zero_()

    def _build_blocks(
        self,
        layer_idx: int,
        block_ids: tuple[int, ...],
        block_keys: torch.Tensor | None = None,
    ) -> None:
        if not block_ids:
            return
        ids, blocks = self._load_block_keys(layer_idx, block_ids, block_keys)
        if blocks.shape[0] != 1:
            raise ValueError("StreamingQueryRobustCache currently requires batch_size=1.")
        vertices = self.query_robust_vertices[layer_idx]
        for start in range(0, int(ids.numel()), self.summary_page_batch):
            stop = min(start + self.summary_page_batch, int(ids.numel()))
            chunk_ids = ids[start:stop]
            # [B,H,P,S,D] -> [P,S,H,D], the QR summary contract.
            keys = blocks[:, :, start:stop].squeeze(0).permute(1, 2, 0, 3).contiguous()
            summary = build_query_robust_page_summaries(
                keys,
                vertices,
                scale=self.query_robust_scale,
                solver_iters=self.solver_iters,
                solver_lr=self.solver_lr,
                uniform_p=self.uniform_p,
            )
            self.landmark_cache[layer_idx].index_copy_(
                1, chunk_ids, summary.landmark.transpose(0, 1)
            )
            self.bias_cache[layer_idx].index_copy_(
                1, chunk_ids, summary.bias.transpose(0, 1)
            )
            self.epsilon_cache[layer_idx].index_copy_(
                1, chunk_ids, summary.epsilon.transpose(0, 1)
            )
            self.duality_gap_cache[layer_idx].index_copy_(
                1, chunk_ids, summary.gap.transpose(0, 1)
            )
            self.metadata_valid[layer_idx].index_fill_(1, chunk_ids, True)

    def _score_blocks(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        first_block: int,
        last_block: int,
    ) -> torch.Tensor:
        if self.incoming_q_len != 1:
            raise ValueError("Query-Robust block routing expects one decode query token.")
        query = query_states.view(
            self.batch_size,
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.incoming_q_len,
            self.head_dim,
        )[:, :, :, -1, :].reshape(
            self.batch_size,
            self.num_attention_heads,
            self.head_dim,
        )
        slots = torch.arange(
            first_block,
            last_block,
            device=self.compute_device,
            dtype=torch.int32,
        ).view(1, -1)
        page_scores = score_query_robust_pages_reference(
            query,
            self.landmark_cache[layer_idx].transpose(0, 1),
            self.bias_cache[layer_idx].transpose(0, 1),
            self.epsilon_cache[layer_idx].transpose(0, 1),
            self.metadata_valid[layer_idx].transpose(0, 1),
            slots,
            scale=self.query_robust_scale,
            alpha=self.score_alpha,
        )
        # QR's reference scorer intentionally chooses one page order per
        # request.  The shared cache interface expects one score per KV head,
        # so broadcast that order without introducing head-specific branches.
        return page_scores[:, None, :].expand(
            self.batch_size, self.num_key_value_heads, -1
        )

    def print_stats(self) -> None:
        super().print_stats()
        valid = int(self.metadata_valid.sum().item())
        print(
            "QUERY_ROBUST_REFERENCE | "
            f"vertices={self.num_vertices} solver_iters={self.solver_iters} "
            f"alpha={self.score_alpha:g} valid_page_heads={valid}"
        )

    @torch.no_grad()
    def query_robust_duality_gap_stats(self) -> dict[str, float | int]:
        gaps = self.duality_gap_cache[self.metadata_valid].float()
        if gaps.numel() == 0:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        quantiles = torch.quantile(gaps, torch.tensor((0.50, 0.95), device=gaps.device))
        return {
            "count": int(gaps.numel()),
            "p50": float(quantiles[0].item()),
            "p95": float(quantiles[1].item()),
            "max": float(gaps.max().item()),
        }


__all__ = ["StreamingQueryRobustCache"]
