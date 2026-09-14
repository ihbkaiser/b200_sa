"""Canonical QUILL scores used only to enrich ShadowKV's block router.

This is deliberately independent of the KVPress runtime.  ShadowKV already
has the three tensors QUILL needs during prefill: pre-RoPE queries, post-RoPE
keys, and values.  Reusing them avoids a second model forward and, more
importantly, guarantees that the routing landmarks and the generation model
see exactly the same tokenisation and hidden states.

The implementation below matches the production QWL-Zeta configuration used
in the paper: D=256, eta=.05, future-RoPE horizon 512, direct adaptive gamma
.05 with [.03,.6] clipping, frozen from the first 1024-token scoring pool, and
Rademacher value projections.  It returns ranks only; it never evicts KV.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _spd_inverse(matrix: torch.Tensor) -> torch.Tensor:
    """Stable batched SPD inverse, matching QUILL's Cholesky path."""
    matrix = 0.5 * (matrix + matrix.transpose(-2, -1))
    eye = torch.eye(
        matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
    ).view(1, 1, matrix.shape[-1], matrix.shape[-1])
    scale = torch.diagonal(matrix, dim1=-2, dim2=-1).abs().amax(
        -1, keepdim=True
    ).unsqueeze(-1).clamp_min(1e-12)
    for jitter in (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        candidate = matrix if jitter == 0.0 else matrix + jitter * scale * eye
        result = torch.linalg.cholesky_ex(candidate)
        if not (result.info != 0).any() and torch.isfinite(result.L).all():
            return torch.cholesky_inverse(result.L)
    raise RuntimeError("QUILL router Cholesky failed after jitter fallback")


def _robust_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    """Scale-aware Cholesky used for the empirical query covariance."""
    matrix = 0.5 * (matrix + matrix.transpose(-2, -1))
    eye = torch.eye(
        matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
    ).view(1, 1, matrix.shape[-1], matrix.shape[-1])
    scale = torch.diagonal(matrix, dim1=-2, dim2=-1).abs().amax(
        -1, keepdim=True
    ).unsqueeze(-1).clamp_min(1e-12)
    for jitter in (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        candidate = matrix if jitter == 0.0 else matrix + jitter * scale * eye
        result = torch.linalg.cholesky_ex(candidate)
        if not (result.info != 0).any() and torch.isfinite(result.L).all():
            return result.L
    raise RuntimeError("QUILL router covariance Cholesky failed after jitter fallback")


class CanonicalQuillRouterScorer:
    """Chunk-local canonical QUILL scorer for exact routing landmarks."""

    def __init__(
        self,
        *,
        n_features: int = 256,
        eta: float = 0.05,
        n_future_positions: int = 512,
        adaptive_direct: float = 0.05,
        gamma_clip_lo: float = 0.03,
        gamma_clip_hi: float = 0.6,
        score_chunk: int = 1024,
        exact_fraction: float = 0.125,
        seed: int = 0,
    ) -> None:
        if not 0.0 < exact_fraction <= 1.0:
            raise ValueError("exact_fraction must lie in (0,1]")
        if score_chunk <= 0:
            raise ValueError("score_chunk must be positive")
        self.n_features = n_features
        self.eta = eta
        self.n_future_positions = n_future_positions
        self.adaptive_direct = adaptive_direct
        self.gamma_clip_lo = gamma_clip_lo
        self.gamma_clip_hi = gamma_clip_hi
        self.score_chunk = score_chunk
        self.exact_fraction = exact_fraction
        self.seed = seed
        self._basis: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _get_basis(
        self,
        layer_idx: int,
        heads: int,
        dim: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (layer_idx, heads, dim, device)
        if key not in self._basis:
            eps_gen = torch.Generator(device="cpu")
            eps_gen.manual_seed(self.seed * 100003 + layer_idx * 1009)
            eps = torch.randn(heads, self.n_features, dim, generator=eps_gen)
            bias = torch.rand(heads, self.n_features, generator=eps_gen) * (
                2.0 * math.pi
            )
            value_gen = torch.Generator(device="cpu")
            value_gen.manual_seed(
                self.seed * 100003 + layer_idx * 1009 + 501
            )
            value_proj = (
                torch.randint(
                    0,
                    2,
                    (heads, self.n_features, dim),
                    generator=value_gen,
                ).float()
                * 2.0
                - 1.0
            )
            self._basis[key] = (
                eps.to(device=device),
                bias.to(device=device),
                value_proj.to(device=device),
            )
        return self._basis[key]

    @staticmethod
    def _spread(
        covariance: torch.Tensor, keys: torch.Tensor
    ) -> torch.Tensor:
        centered = keys.float() - keys.float().mean(dim=2, keepdim=True)
        key_covariance = torch.einsum(
            "bhtd,bhte->bhde", centered, centered
        ) / max(keys.shape[2], 1)
        return torch.einsum(
            "bhde,bhed->bh", covariance / float(keys.shape[-1]), key_covariance
        ).clamp_min(1e-9)

    def _scores_one_pool(
        self,
        *,
        layer_idx: int,
        queries_prerope: torch.Tensor,
        keys_postrope: torch.Tensor,
        values: torch.Tensor,
        next_position: int,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        gamma_store: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        batch, kv_heads, tokens, dim = keys_postrope.shape
        query_heads = queries_prerope.shape[1]
        groups = query_heads // kv_heads

        future = torch.arange(
            next_position,
            next_position + self.n_future_positions,
            device=keys_postrope.device,
        )
        mean_cos = cos_cache[future].float().mean(dim=0)
        mean_sin = sin_cache[future].float().mean(dim=0)
        rotated_queries = (
            queries_prerope.float() * mean_cos
            + _rotate_half(queries_prerope.float()) * mean_sin
        )
        q_kv = rotated_queries.view(
            batch, kv_heads, groups, tokens, dim
        ).mean(dim=2)
        q_centered = q_kv - q_kv.mean(dim=2, keepdim=True)
        covariance = torch.einsum(
            "bhtd,bhte->bhde", q_centered, q_centered
        ) / max(tokens, 1)

        if layer_idx not in gamma_store:
            spread = self._spread(covariance, keys_postrope)
            gamma_store[layer_idx] = (
                self.adaptive_direct * spread
            ).clamp(self.gamma_clip_lo, self.gamma_clip_hi).detach()
        gamma = gamma_store[layer_idx].to(covariance.device).view(
            batch, kv_heads, 1, 1
        )
        metric = covariance * gamma
        diag_mean = torch.diagonal(metric, dim1=-2, dim2=-1).mean(
            -1, keepdim=True
        ).unsqueeze(-1)
        eye_dim = torch.eye(
            dim, device=metric.device, dtype=metric.dtype
        ).view(1, 1, dim, dim)
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        metric = metric + diag_mean.clamp_min(1e-12) * 1e-5 * eye_dim
        chol = _robust_cholesky(metric / float(dim))

        eps, bias, value_proj = self._get_basis(
            layer_idx, kv_heads, dim, keys_postrope.device
        )
        transformed_keys = torch.einsum(
            "bhij,bhtj->bhti", chol.transpose(-2, -1), keys_postrope.float()
        )
        phase = (
            torch.einsum("bhtd,hkd->bhtk", transformed_keys, eps)
            + bias.view(1, kv_heads, 1, self.n_features)
        )
        projected_values = torch.einsum(
            "bhtd,hkd->bhtk", values.float(), value_proj
        )
        features = (
            math.sqrt(2.0 / self.n_features)
            * torch.cos(phase)
            * projected_values
        )
        gram = torch.einsum("bhtd,bhte->bhde", features, features)
        gram = 0.5 * (gram + gram.transpose(-2, -1))
        ridge = (
            self.eta
            * features.square().sum(dim=(-2, -1))
            / max(tokens, 1)
        ).view(batch, kv_heads, 1, 1)
        eye_feature = torch.eye(
            self.n_features, device=gram.device, dtype=gram.dtype
        ).view(1, 1, self.n_features, self.n_features)
        inverse = _spd_inverse(gram + ridge * eye_feature)
        projected = torch.einsum("bhde,bhte->bhtd", inverse, features)
        scores = (features * projected).sum(dim=-1)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("non-finite canonical QUILL router scores")
        return scores

    @torch.inference_mode()
    def exact_mask(
        self,
        *,
        layer_idx: int,
        queries_prerope: torch.Tensor,
        keys_postrope: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        gamma_store: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Return [B,H_kv,T] exact candidates, independently per score pool."""
        batch, kv_heads, total, _ = keys_postrope.shape
        mask = torch.zeros(
            batch, kv_heads, total, dtype=torch.bool, device=keys_postrope.device
        )
        for start in range(0, total, self.score_chunk):
            stop = min(start + self.score_chunk, total)
            scores = self._scores_one_pool(
                layer_idx=layer_idx,
                queries_prerope=queries_prerope[:, :, start:stop],
                keys_postrope=keys_postrope[:, :, start:stop],
                values=values[:, :, start:stop],
                next_position=int(position_ids[0, stop - 1]) + 1,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                gamma_store=gamma_store,
            )
            count = max(1, int(math.ceil(self.exact_fraction * (stop - start))))
            chosen = scores.topk(count, dim=-1).indices + start
            mask.scatter_(2, chosen, True)
        return mask


class KeyDiffRouterScorer:
    """Canonical KVPress KeyDiff ranking, independently per score pool."""

    def __init__(self, *, score_chunk: int = 1024, exact_fraction: float = 0.125):
        if score_chunk <= 0:
            raise ValueError("score_chunk must be positive")
        if not 0.0 < exact_fraction <= 1.0:
            raise ValueError("exact_fraction must lie in (0,1]")
        self.score_chunk = score_chunk
        self.exact_fraction = exact_fraction

    @torch.inference_mode()
    def exact_mask(
        self,
        *,
        keys_postrope: torch.Tensor,
        **_ignored,
    ) -> torch.Tensor:
        """Match KeyDiffPress: -cos(k_i, mean_j normalize(k_j))."""
        batch, heads, total, _ = keys_postrope.shape
        mask = torch.zeros(
            batch, heads, total, dtype=torch.bool, device=keys_postrope.device
        )
        for start in range(0, total, self.score_chunk):
            stop = min(start + self.score_chunk, total)
            keys = keys_postrope[:, :, start:stop]
            anchor = F.normalize(keys, p=2, dim=-1).mean(dim=2, keepdim=True)
            score = -F.cosine_similarity(keys, anchor, dim=-1)
            count = max(1, int(math.ceil(self.exact_fraction * (stop - start))))
            chosen = score.topk(count, dim=-1).indices + start
            mask.scatter_(2, chosen, True)
        return mask


def combine_mean_and_exact_logits(
    mean_logits: torch.Tensor,
    exact_logits: torch.Tensor | None,
    exact_block_indices: torch.Tensor | None,
    exact_valid: torch.Tensor | None,
    aggregation: str = "max",
) -> torch.Tensor:
    """Max-pool exact token logits into their blocks and meet with the mean.

    Shapes are mean [B,H,G,Q,C], exact [B,H,G,Q,E], indices/valid [B,H,E].
    The helper is separate so the routing rule can be unit-tested without a
    model or CUDA kernel.
    """
    if exact_logits is None or exact_logits.shape[-1] == 0:
        return mean_logits
    index = exact_block_indices[:, :, None, None, :].expand_as(exact_logits)
    valid = exact_valid[:, :, None, None, :].expand_as(exact_logits)
    source = exact_logits.masked_fill(~valid, float("-inf"))
    if aggregation == "logsumexp":
        source32 = source.float()
        block_max = torch.full_like(
            mean_logits, float("-inf"), dtype=torch.float32
        )
        block_max.scatter_reduce_(
            dim=-1, index=index, src=source32, reduce="amax"
        )
        selected_max = block_max.gather(dim=-1, index=index)
        stable_exp = torch.where(
            valid,
            torch.exp(source32 - selected_max),
            torch.zeros((), device=source.device, dtype=torch.float32),
        )
        block_sum = torch.zeros_like(block_max)
        block_sum.scatter_add_(dim=-1, index=index, src=stable_exp)
        has_exact = block_sum > 0
        block_lse = block_max + torch.log(block_sum.clamp_min(1e-30))
        return torch.where(
            has_exact, block_lse.to(mean_logits.dtype), mean_logits
        )
    if aggregation != "max":
        raise ValueError(f"unknown router aggregation: {aggregation}")
    pooled = torch.full_like(mean_logits, float("-inf"))
    pooled.scatter_reduce_(dim=-1, index=index, src=source, reduce="amax")
    return torch.maximum(mean_logits, pooled)
