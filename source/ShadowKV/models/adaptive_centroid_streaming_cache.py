"""Standalone streaming form of the adaptive centroid/LSE router.

Unlike the historical experiment class, this cache does not inherit released
ShadowKV and therefore does not rely on a prefill-only SVD. It stores exact
post-RoPE KV, either on GPU or in pinned CPU memory, and incrementally indexes
every newly sealed block. The default quality path uses PyTorch; the optional
two-slot Triton backend fuses decode routing without changing the representation
or allocator.
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn

from .adaptive_centroid_incremental import IncrementalTwoCentroidGraph
from .adaptive_centroid_triton import packed_block_logits, two_slot_block_logits
from .centroid_router_cache import (
    allocate_concave_marginal_counts,
    allocate_minimax_centroid_counts,
    allocate_rate_distortion_counts,
    allocate_distortion_target_counts,
    demand_adaptive_rate_distortion_counts,
    hierarchical_log_rate_distortion_counts,
    evaluate_proxy_lse_gap_path,
    evaluate_self_lse_cvar_path,
    distortion_threshold_counts,
    fit_agglomerative_self_lse_paths,
    fit_angular_isolation_lse_paths,
    fit_lazy_exact_self_lse_allocation,
    fit_observed_query_lse_paths,
    fit_observed_query_mass_paths,
    fit_observed_query_trimmed_cvar_paths,
    fit_residual_isolation_lse_paths,
    fit_residual_tail_lse_paths,
    fit_residual_tail_path,
    fit_value_tail_path,
    priced_concave_marginal_counts,
    rate_distortion_counts,
    relative_rate_distortion_counts,
)
from .streaming_cache import StreamingBlockCache


@torch.inference_mode()
def evaluate_angular_radius_path(
    keys: torch.Tensor,
    assignments: torch.Tensor,
    batch_blocks: int = 512,
) -> torch.Tensor:
    """Worst angular residual of every nested component path.

    ``assignments[..., r - 1, :]`` partitions an ``S``-token block into
    ``r`` components.  For every token we measure one minus its cosine with
    the normalized mean direction of its assigned component, then retain the
    largest residual in the block.  This is a query-free, scale-invariant
    measure of whether one directional summary dilutes an angular outlier.

    The block loop bounds temporary storage at long context.  ``cummin``
    makes the reported distortion path nested even when a placement path was
    optimized for a different objective.
    """
    if keys.ndim != 5 or assignments.ndim != 5:
        raise ValueError("invalid key or assignment rank")
    batch, heads, n_blocks, block_size, _ = keys.shape
    if assignments.shape[:3] != (batch, heads, n_blocks):
        raise ValueError("assignment geometry does not match keys")
    stages = assignments.shape[-2]
    if assignments.shape[-1] != block_size or stages > block_size:
        raise ValueError("invalid assignment path geometry")
    if batch_blocks < 1:
        raise ValueError("batch_blocks must be positive")

    risk = torch.empty(
        batch, heads, n_blocks, stages,
        device=keys.device, dtype=torch.float32,
    )
    for start in range(0, n_blocks, batch_blocks):
        stop = min(start + batch_blocks, n_blocks)
        unit = nn.functional.normalize(
            keys[..., start:stop, :, :].float(), dim=-1, eps=1e-12
        )
        for r in range(1, stages + 1):
            labels = assignments[..., start:stop, r - 1, :].long()
            membership = nn.functional.one_hot(
                labels, num_classes=r
            ).to(unit.dtype)
            sums = torch.einsum(
                "bhnsr,bhnsd->bhnrd", membership, unit
            )
            centers = nn.functional.normalize(sums, dim=-1, eps=1e-12)
            cosine = torch.einsum(
                "bhnsd,bhnrd->bhnsr", unit, centers
            )
            selected = cosine.gather(-1, labels[..., None]).squeeze(-1)
            risk[..., start:stop, r - 1] = (
                1.0 - selected.clamp(-1.0, 1.0)
            ).amax(-1)
    risk[..., -1] = 0
    return torch.cummin(risk, dim=-1).values


@torch.inference_mode()
def estimate_self_k_block_exposure(
    keys: torch.Tensor,
    reference_count: int = 64,
    temperature: float = 1.0,
    batch_blocks: int = 512,
) -> torch.Tensor:
    """Query-free block exposure under an empirical self-K direction bank.

    Uniformly sampled normalized key directions stand in for unknown future
    query directions.  For every direction we compute the *exact* block mass
    and normalize it across the full prompt.  Averaging those probabilities
    estimates how often a block is addressable, while normalization to unit
    mean makes the result a pure allocation weight.  No task labels or decode
    queries enter this estimate.
    """
    if keys.ndim != 5:
        raise ValueError("keys must have shape [B,H,N,S,D]")
    if reference_count < 1 or temperature <= 0 or batch_blocks < 1:
        raise ValueError("invalid self-K exposure configuration")
    _, _, n_blocks, block_size, _ = keys.shape
    flat = keys.flatten(-3, -2)
    count = min(reference_count, flat.shape[-2])
    index = torch.linspace(
        0, flat.shape[-2] - 1, count, device=keys.device
    ).round().long().unique()
    reference = nn.functional.normalize(
        flat.index_select(-2, index).float(), dim=-1, eps=1e-12
    )
    pieces = []
    for start in range(0, n_blocks, batch_blocks):
        stop = min(start + batch_blocks, n_blocks)
        logits = torch.einsum(
            "bhqd,bhnsd->bhqns",
            reference,
            keys[..., start:stop, :, :].float(),
        )
        pieces.append(torch.logsumexp(temperature * logits, dim=-1))
    block_log_mass = torch.cat(pieces, dim=-1)
    probability = torch.softmax(block_log_mass, dim=-1).mean(dim=-2)
    return probability / probability.mean(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.inference_mode()
def _macro_angular_kde_maxmed_signal(
    keys: torch.Tensor,
    page_blocks: int,
    reference_count: int,
    kappa: float,
    signal_mode: str = "contrast",
) -> torch.Tensor:
    """Per-block angular tail contrast from query-free macro K context."""
    if page_blocks < 1 or reference_count < 2 or kappa <= 0:
        raise ValueError("invalid angular-density allocation geometry")
    if signal_mode not in {"contrast", "absolute_contrast", "absolute_max"}:
        raise ValueError("unknown angular-density block signal")
    batch, heads, n_blocks, block_size, _ = keys.shape
    pieces = []
    for start in range(0, n_blocks, page_blocks):
        stop = min(start + page_blocks, n_blocks)
        page = keys[..., start:stop, :, :].float()
        unit = nn.functional.normalize(
            page.flatten(-3, -2), dim=-1, eps=1e-12
        )
        token_count = unit.shape[-2]
        count = min(reference_count, token_count)
        index = torch.linspace(
            0, token_count - 1, count, device=keys.device
        ).round().long().unique()
        reference = unit[..., index, :]
        cosine = torch.matmul(unit, reference.transpose(-1, -2))
        cosine[
            ..., index, torch.arange(index.numel(), device=keys.device)
        ] = -torch.inf
        log_density = torch.logsumexp(kappa * cosine, dim=-1)
        log_density = log_density - math.log(max(1, index.numel() - 1))
        inverse = torch.exp(-log_density).view(
            batch, heads, stop - start, block_size
        )
        contrast = inverse.amax(-1) / inverse.median(-1).values.clamp_min(1e-6)
        if signal_mode == "absolute_contrast":
            # Page-normalized absolute rarity restores the macro role erased
            # by block normalization; contrast retains the local dilution
            # signal.  Their product is relevance times representational need.
            absolute = inverse.mean(-1)
            absolute = absolute / absolute.mean(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            signal = absolute * contrast
        elif signal_mode == "absolute_max":
            # Expected relevance under a smooth angular query law is governed
            # by exposed directions, so use the rarest member of the block.
            # Unlike absolute_contrast this does not multiply two correlated
            # tail statistics; it was the most robust trace-level signal and
            # preserved oracle answer mass on the CWE diagnostic.
            signal = inverse.amax(-1)
            signal = signal / signal.mean(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
        else:
            signal = contrast
        pieces.append(signal)
    return torch.cat(pieces, dim=-1)


@torch.inference_mode()
def _head_trusted_density_weight(
    normalized_signal: torch.Tensor,
    risk: torch.Tensor,
    shrinkage: float,
    power: float,
    up_only: bool = False,
) -> torch.Tensor:
    """Shrink the macro-density prior on locally fragile KV heads.

    ``risk[..., b, r-1]`` is the local distortion after retaining ``r``
    components.  The upper-tail 1->2 gain is therefore a query-free measure
    of how costly a one-center approximation can be for a head.  We compare
    it with the median head in the same batch and smoothly reduce the density
    prior, rather than making a hard semantic-head decision.
    """
    if power < 0:
        raise ValueError("density head-trust power must be non-negative")
    departure = normalized_signal - 1.0
    if up_only:
        departure = departure.clamp_min(0)
    if power == 0:
        return 1.0 + shrinkage * departure
    gain12 = (risk[..., 0] - risk[..., 1]).clamp_min(0)
    sensitivity = torch.quantile(
        gain12.float(), 0.90, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    reference = sensitivity.median(dim=1, keepdim=True).values
    trust = (reference / sensitivity).clamp(max=1.0).pow(power)
    return 1.0 + shrinkage * trust.to(departure.dtype) * departure


def _quantize_centers(
    centers: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Symmetric per-center quantization; INT4 is physically nibble-packed."""
    if bits == 16:
        return centers, None
    qmax = 127 if bits == 8 else 7
    scale = centers.float().abs().amax(-1).div(qmax).clamp_min(1e-12)
    quantized = torch.round(centers.float() / scale[..., None]).clamp(
        -qmax, qmax
    ).to(torch.int8)
    if bits == 8:
        return quantized, scale
    if centers.shape[-1] % 2:
        quantized = nn.functional.pad(quantized, (0, 1))
    unsigned = quantized.to(torch.int16).bitwise_and(0xF).to(torch.uint8)
    packed = unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)
    return packed.contiguous(), scale


def _dequantize_centers(
    stored: torch.Tensor,
    scale: torch.Tensor | None,
    bits: int,
    dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if bits == 16:
        return stored.to(dtype)
    if scale is None:
        raise RuntimeError("quantized centers are missing their scale")
    if bits == 8:
        quantized = stored
    else:
        low = stored & 0x0F
        high = (stored >> 4) & 0x0F
        nibble = torch.stack((low, high), dim=-1).reshape(
            *stored.shape[:-1], -1
        )[..., :dim]
        quantized = torch.where(
            nibble >= 8, nibble.to(torch.int16) - 16, nibble.to(torch.int16)
        )
    return (quantized.float() * scale.float()[..., None]).to(dtype)


def _matrix_root(
    matrix: torch.Tensor, *, inverse: bool = False
) -> torch.Tensor:
    """Symmetric PSD square root used by the query-aware rank sketch."""
    value, vector = torch.linalg.eigh(0.5 * (matrix + matrix.T))
    cutoff = value.amax().clamp_min(1e-30) * 1e-7
    if inverse:
        diagonal = torch.where(
            value > cutoff, value.rsqrt(), torch.zeros_like(value)
        )
    else:
        diagonal = value.clamp_min(0).sqrt()
    return (vector * diagonal[None]) @ vector.T


def _optimal_bilinear_factors(
    query_second_moment: torch.Tensor,
    key_second_moment: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor the minimum-MSE rank-r approximation q^T A B^T k."""
    query_root = _matrix_root(query_second_moment)
    key_root = _matrix_root(key_second_moment)
    left, singular, right_h = torch.linalg.svd(
        query_root @ key_root, full_matrices=False
    )
    weight = singular[:rank].clamp_min(0).sqrt()
    query_factor = (
        _matrix_root(query_second_moment, inverse=True)
        @ (left[:, :rank] * weight[None])
    )
    key_factor = (
        _matrix_root(key_second_moment, inverse=True)
        @ (right_h[:rank].T * weight[None])
    )
    return query_factor, key_factor


def _key_pca_factors(
    key_second_moment: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K-only PCA control analogous to ShadowKV's truncated K SVD.

    Using the same orthonormal factor on both sides realizes
    ``q^T V_r V_r^T k`` without reconstructing the full-dimensional key.
    """
    value, vector = torch.linalg.eigh(
        0.5 * (key_second_moment + key_second_moment.T)
    )
    factor = vector[:, -rank:].flip(-1).contiguous()
    return factor, factor


@torch.inference_mode()
def _padded_adaptive_centroids(
    keys: torch.Tensor,
    assignment_path: torch.Tensor,
    component_counts: torch.Tensor,
    slot_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize mixtures into the cache's bounded dense slot count."""
    block_size = keys.shape[-2]
    if int(component_counts.max()) > slot_count:
        raise ValueError("component count exceeds allocated router slots")
    chosen = assignment_path.gather(
        -2,
        (component_counts - 1)[..., None, None].expand(
            *component_counts.shape, 1, block_size
        ),
    ).squeeze(-2).long()
    membership = nn.functional.one_hot(
        chosen, num_classes=block_size
    ).float()[..., :slot_count]
    populations = membership.sum(-2)
    sums = torch.einsum("...nsc,...nsd->...ncd", membership, keys.float())
    centers = sums / populations.clamp_min(1)[..., None]
    energy = torch.einsum(
        "...nsc,...ns->...nc",
        membership,
        keys.float().square().sum(-1),
    )
    trace = (
        energy / populations.clamp_min(1)
        - centers.square().sum(-1)
    ).clamp_min_(0)
    alpha = trace / (2.0 * keys.shape[-1])
    valid = (
        torch.arange(slot_count, device=keys.device)
        < component_counts[..., None]
    )
    log_count = torch.where(
        valid,
        populations.clamp_min(1).log(),
        torch.full_like(populations, float("-inf")),
    )
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
    centers = torch.where(valid[..., None], centers, torch.zeros_like(centers))
    return centers.to(keys.dtype), log_count.to(keys.dtype), alpha


@torch.inference_mode()
def _padded_component_value_means(
    values: torch.Tensor,
    assignment_path: torch.Tensor,
    component_counts: torch.Tensor,
    slot_count: int,
) -> torch.Tensor:
    """Mean value vector for each fitted key component.

    The key partition already exists for routing.  Reusing exactly that
    membership on V preserves the joint K/V mixture seen by the attention
    output, instead of collapsing a multi-directional block to one mean V.
    """
    block_size = values.shape[-2]
    chosen = assignment_path.gather(
        -2,
        (component_counts - 1)[..., None, None].expand(
            *component_counts.shape, 1, block_size
        ),
    ).squeeze(-2).long()
    membership = nn.functional.one_hot(
        chosen, num_classes=block_size
    ).float()[..., :slot_count]
    populations = membership.sum(-2).clamp_min(1)
    means = torch.einsum(
        "...nsc,...nsd->...ncd", membership, values.float()
    ) / populations[..., None]
    valid = (
        torch.arange(slot_count, device=values.device)
        < component_counts[..., None]
    )
    return torch.where(
        valid[..., None], means, torch.zeros_like(means)
    ).to(values.dtype)


class StreamingAdaptiveCentroidLSECache(StreamingBlockCache):
    """Adaptive variable-order centroid/LSE routing over a growing index."""

    def __init__(
        self,
        config: object,
        *,
        extra_fraction: float | None = None,
        self_lse_temperatures: tuple[float, ...] = (1.0,),
        query_group_mean: bool = False,
        router_backend: str = "torch",
        refine_factor: float = 1.0,
        refine_candidate_ratio: float | None = None,
        refine_tokens: bool = False,
        max_components: int | None = None,
        compact_metadata: bool = False,
        center_bits: int = 16,
        **kwargs,
    ) -> None:
        block_size = int(kwargs.get("block_size", 0))
        if not block_size:
            raise ValueError("streaming adaptive router requires block_size")
        if extra_fraction is None:
            # The query-mean default uses an unconstrained, fixed-price
            # rate--distortion decision; it has no global center quota.
            extra_fraction = 0.0 if query_group_mean else 0.25
        if not 0.0 <= extra_fraction <= block_size - 1:
            raise ValueError(
                "streaming extra_fraction must lie in "
                f"[0,{block_size - 1}]"
            )
        if not self_lse_temperatures or any(
            temperature <= 0 for temperature in self_lse_temperatures
        ):
            raise ValueError("self-LSE temperatures must be positive")
        self.extra_fraction = float(extra_fraction)
        self.self_lse_temperatures = tuple(self_lse_temperatures)
        # The three router knobs below were an experiment surface: 43 environment
        # variables selecting among 45 code paths, of which the campaign ever ran
        # two. They are constructor state now, so a stale shell variable can no
        # longer change the method while the cell name says otherwise.
        self.self_lse_cost = "mean_gap" if query_group_mean else "max_gap"
        self.self_lse_cost_beta = float(os.environ.get(
            "SHADOWKV_SELF_LSE_COST_BETA", "4"
        ))
        if self.self_lse_cost_beta <= 0:
            raise ValueError("self-LSE cost beta must be positive")
        correction = os.environ.get(
            "SHADOWKV_CENTER_DISPERSION_CORRECTION",
            "0" if query_group_mean else "1",
        ).strip()
        if correction not in {"0", "1"}:
            raise ValueError(
                "SHADOWKV_CENTER_DISPERSION_CORRECTION must be 0 or 1"
            )
        self.center_dispersion_correction = correction == "1"
        self.center_placement = "self"
        self.center_allocation = (
            "tail_absolute_rate_distortion" if query_group_mean else "self_lse"
        )
        if (
            self.center_allocation == "residual_radius"
            and self.center_placement != "self"
        ):
            raise ValueError(
                "residual-radius allocation currently requires self placement"
            )
        if (
            self.center_allocation.startswith("residual_")
            and self.center_allocation != "residual_radius"
            and self.center_placement != "residual_path"
        ):
            raise ValueError(
                "residual allocation requires residual_path placement"
            )
        if (
            self.center_allocation == "anchor_tail_cvar"
            and self.center_placement not in {"self", "self_qanchor"}
        ):
            raise ValueError(
                "anchor-tail allocation requires self or self_qanchor placement"
            )
        self.density_page_tokens = int(os.environ.get(
            "SHADOWKV_DENSITY_PAGE_TOKENS", "1024"
        ))
        self.density_reference_count = int(os.environ.get(
            "SHADOWKV_DENSITY_REFERENCE_COUNT", "64"
        ))
        self.exposure_reference_count = int(os.environ.get(
            "SHADOWKV_EXPOSURE_REFERENCE_COUNT", "64"
        ))
        self.density_kappa = float(os.environ.get(
            "SHADOWKV_DENSITY_KAPPA", "8"
        ))
        self.density_shrinkage = float(os.environ.get(
            "SHADOWKV_DENSITY_SHRINKAGE", "0.5"
        ))
        self.density_head_trust_power = float(os.environ.get(
            "SHADOWKV_DENSITY_HEAD_TRUST_POWER", "0"
        ))
        self.density_value_power = float(os.environ.get(
            "SHADOWKV_DENSITY_VALUE_POWER", "0"
        ))
        self.density_weight_mode = os.environ.get(
            "SHADOWKV_DENSITY_WEIGHT_MODE", "symmetric"
        ).strip().lower()
        if self.density_weight_mode not in {"symmetric", "up_only"}:
            raise ValueError(
                "SHADOWKV_DENSITY_WEIGHT_MODE must be symmetric or up_only"
            )
        self.density_signal_mode = os.environ.get(
            "SHADOWKV_DENSITY_SIGNAL_MODE", "contrast"
        ).strip().lower()
        if self.density_signal_mode not in {
            "contrast", "absolute_contrast", "absolute_max"
        }:
            raise ValueError(
                "SHADOWKV_DENSITY_SIGNAL_MODE must be contrast or "
                "absolute_contrast or absolute_max"
            )
        if self.density_page_tokens % block_size:
            raise ValueError("density page tokens must align to block size")
        if self.density_reference_count < 2 or self.density_kappa <= 0:
            raise ValueError("invalid density reference count or kappa")
        if self.exposure_reference_count < 1:
            raise ValueError("exposure reference count must be positive")
        if not 0.0 <= self.density_shrinkage <= 1.0:
            raise ValueError("density shrinkage must lie in [0,1]")
        if self.density_head_trust_power < 0:
            raise ValueError("density head-trust power must be non-negative")
        if self.density_value_power < 0:
            raise ValueError("density value power must be non-negative")
        self.query_group_mean = bool(query_group_mean)
        self.query_mass_bank_size = int(os.environ.get(
            "SHADOWKV_QUERY_MASS_BANK_SIZE", "32"
        ))
        self.query_mass_window = int(os.environ.get(
            "SHADOWKV_QUERY_MASS_WINDOW", "512"
        ))
        self.query_mass_objective = os.environ.get(
            "SHADOWKV_QUERY_MASS_OBJECTIVE", "global_mass"
        ).strip().lower()
        self.query_mass_gqa_reduce = os.environ.get(
            "SHADOWKV_QUERY_MASS_GQA_REDUCE", "mean"
        ).strip().lower()
        if self.query_mass_bank_size < 1 or self.query_mass_window < 1:
            raise ValueError("query-mass bank size and window must be positive")
        if self.query_mass_objective not in {
            "global_mass", "global_mass_cvar", "topk_hinge",
            "topk_hinge_cvar",
        }:
            raise ValueError(
                "SHADOWKV_QUERY_MASS_OBJECTIVE must be global_mass, "
                "global_mass_cvar, topk_hinge, or topk_hinge_cvar"
            )
        if self.query_mass_gqa_reduce not in {"mean", "all"}:
            raise ValueError(
                "SHADOWKV_QUERY_MASS_GQA_REDUCE must be mean or all"
            )
        self.robust_query_source = os.environ.get(
            "SHADOWKV_ROBUST_QUERY_SOURCE", "self_k"
        ).strip().lower()
        if self.robust_query_source not in {"self_k", "final"}:
            raise ValueError(
                "SHADOWKV_ROBUST_QUERY_SOURCE must be self_k or final"
            )
        if (
            self.robust_query_source == "final"
            and self.center_placement != "robust_trimmed"
        ):
            raise ValueError(
                "final robust proxy requires robust_trimmed placement"
            )
        self.robust_trim_fraction = float(os.environ.get(
            "SHADOWKV_ROBUST_TRIM_FRACTION", "0.25"
        ))
        if not 0.0 <= self.robust_trim_fraction < 1.0:
            raise ValueError("robust trim fraction must lie in [0,1)")
        self.tail_cvar_fraction = float(os.environ.get(
            "SHADOWKV_TAIL_CVAR_FRACTION", "0.25"
        ))
        if not 0.0 < self.tail_cvar_fraction <= 1.0:
            raise ValueError("tail CVaR fraction must lie in (0,1]")
        self.self_lse_batch_blocks = int(os.environ.get(
            "SHADOWKV_SELF_LSE_BATCH_BLOCKS", "65536"
        ))
        if self.self_lse_batch_blocks < 1:
            raise ValueError("self-LSE batch size must be positive")
        self.relative_rd_penalty = float(os.environ.get(
            "SHADOWKV_RELATIVE_RD_PENALTY", "0.25"
        ))
        if self.relative_rd_penalty < 0:
            raise ValueError(
                "relative rate-distortion penalty must be non-negative"
            )
        default_absolute_rd_penalty = "1.5" if query_group_mean else "0.25"
        self.absolute_rd_penalty = float(os.environ.get(
            "SHADOWKV_ABSOLUTE_RD_PENALTY", default_absolute_rd_penalty
        ))
        if self.absolute_rd_penalty < 0:
            raise ValueError(
                "absolute rate-distortion penalty must be non-negative"
            )
        self.distortion_tolerance = float(os.environ.get(
            "SHADOWKV_DISTORTION_TOLERANCE", "1.0"
        ))
        if self.distortion_tolerance < 0:
            raise ValueError("distortion tolerance must be non-negative")
        self.group_distortion_target = float(os.environ.get(
            "SHADOWKV_GROUP_DISTORTION_TARGET", "1.0"
        ))
        if self.group_distortion_target < 0:
            raise ValueError("group distortion target must be non-negative")
        self.demand_low_penalty = float(os.environ.get(
            "SHADOWKV_DEMAND_LOW_PENALTY", "1.5"
        ))
        self.demand_high_penalty = float(os.environ.get(
            "SHADOWKV_DEMAND_HIGH_PENALTY", "2.5"
        ))
        self.demand_threshold = float(os.environ.get(
            "SHADOWKV_DEMAND_THRESHOLD", "1.75"
        ))
        if not 0 <= self.demand_low_penalty <= self.demand_high_penalty:
            raise ValueError("demand penalties must satisfy 0 <= low <= high")
        if self.demand_threshold < 1:
            raise ValueError("demand threshold must be at least one")
        self.hierarchical_rd_alpha = float(os.environ.get(
            "SHADOWKV_HIERARCHICAL_RD_ALPHA", "3.0"
        ))
        self.hierarchical_rd_beta = float(os.environ.get(
            "SHADOWKV_HIERARCHICAL_RD_BETA", "1.0"
        ))
        self.hierarchical_rd_group = os.environ.get(
            "SHADOWKV_HIERARCHICAL_RD_GROUP", "head"
        ).strip().lower()
        if self.hierarchical_rd_alpha < 0:
            raise ValueError("hierarchical RD alpha must be non-negative")
        if self.hierarchical_rd_beta <= 0:
            raise ValueError("hierarchical RD beta must be positive")
        if self.hierarchical_rd_group not in {"head", "layer"}:
            raise ValueError("hierarchical RD group must be head or layer")
        self.hierarchical_rd_base = float(os.environ.get(
            "SHADOWKV_HIERARCHICAL_RD_BASE", "1.5"
        ))
        if self.hierarchical_rd_base < 0:
            raise ValueError("hierarchical RD base must be non-negative")
        self.qanchor_weight = float(os.environ.get(
            "SHADOWKV_QANCHOR_WEIGHT", "0.25"
        ))
        if not 0.0 <= self.qanchor_weight <= 1.0:
            raise ValueError("query-anchor weight must lie in [0,1]")
        # The canonical mean-gap objective needs no empirical dispersion
        # correction. Historical ablations can still opt in explicitly.
        default_gap_correction = "0"
        self.tail_gap_correction_scale = float(os.environ.get(
            "SHADOWKV_TAIL_GAP_CORRECTION_SCALE", default_gap_correction
        ))
        if self.tail_gap_correction_scale < 0:
            raise ValueError("tail-gap correction scale must be non-negative")
        if refine_factor < 1.0:
            raise ValueError("refine_factor must be at least one")
        self.refine_factor = float(refine_factor)
        if refine_candidate_ratio is not None and not (
            0.0 < refine_candidate_ratio <= 1.0
        ):
            raise ValueError("refine_candidate_ratio must lie in (0,1]")
        self.refine_candidate_ratio = (
            None if refine_candidate_ratio is None
            else float(refine_candidate_ratio)
        )
        self.refine_tokens = bool(refine_tokens)
        self.component_candidate_refinement = (
            os.environ.get("STREAMING_REFINE_COMPONENT_CANDIDATES", "0") == "1"
        )
        self.match_token_allocation = (
            os.environ.get("STREAMING_REFINE_MATCH_ALLOCATION", "0") == "1"
        )
        self.refine_token_gqa_reduce = os.environ.get(
            "STREAMING_REFINE_TOKEN_GQA_REDUCE", "mean"
        ).strip().lower()
        if self.refine_token_gqa_reduce not in {
            "mean", "p2", "p4", "ucb", "max"
        }:
            raise ValueError(
                "token-refinement GQA reduction must be mean, p2, p4, ucb, or max"
            )
        self.refine_token_gqa_ucb = float(os.environ.get(
            "STREAMING_REFINE_TOKEN_GQA_UCB", "1.0"
        ))
        if self.refine_token_gqa_ucb < 0:
            raise ValueError("token-refinement GQA UCB weight must be nonnegative")
        if self.match_token_allocation and not self.refine_tokens:
            raise ValueError(
                "matched token allocation requires token refinement"
            )
        self.component_vmean_allocation = (
            os.environ.get("SHADOWKV_COMPONENT_VMEAN_ALLOCATION", "0") == "1"
        )
        self.store_router_value_mean = (
            os.environ.get("SHADOWKV_STORE_ROUTER_VALUE_MEAN", "0") == "1"
            or self.component_vmean_allocation
            or self.center_placement == "value_tail"
            or (
                self.center_allocation == "density_tail_cvar"
                and self.density_value_power > 0
            )
        )
        self.component_vmean_blend = float(os.environ.get(
            "SHADOWKV_COMPONENT_VMEAN_BLEND", "1.0"
        ))
        if not 0.0 <= self.component_vmean_blend <= 1.0:
            raise ValueError("component-V blend must lie in [0,1]")
        if self.refine_tokens and (
            self.refine_candidate_ratio is None and self.refine_factor == 1.0
        ):
            raise ValueError("token refinement requires a candidate expansion")
        self.refine_sketch_rank = int(os.environ.get(
            "STREAMING_REFINE_SKETCH_RANK", "0"
        ))
        self.refine_sketch_bits = int(os.environ.get(
            "STREAMING_REFINE_SKETCH_BITS", "4"
        ))
        self.refine_sketch_prompt_samples = int(os.environ.get(
            "STREAMING_REFINE_SKETCH_PROMPT_SAMPLES", "512"
        ))
        self.refine_sketch_basis = os.environ.get(
            "STREAMING_REFINE_SKETCH_BASIS", "query_aware"
        )
        configured_head_dim = int(getattr(
            config, "head_dim",
            config.hidden_size // config.num_attention_heads,
        ))
        if not 0 <= self.refine_sketch_rank <= configured_head_dim:
            raise ValueError("refinement sketch rank must lie in [0, head_dim]")
        if self.refine_sketch_bits not in {4, 8, 16}:
            raise ValueError("refinement sketch bits must be 4, 8, or 16")
        if self.refine_sketch_prompt_samples < 1:
            raise ValueError("refinement prompt sample count must be positive")
        if self.refine_sketch_basis not in {"query_aware", "key_pca"}:
            raise ValueError(
                "refinement sketch basis must be query_aware or key_pca"
            )
        if self.refine_sketch_rank and not self.refine_tokens:
            raise ValueError("rank-sketch refinement requires token refinement")
        if self.component_candidate_refinement and not self.refine_tokens:
            raise ValueError("component candidates require token refinement")
        if self.component_candidate_refinement and compact_metadata:
            raise ValueError(
                "the exact component-candidate diagnostic requires dense metadata"
            )
        # Token refinement returns individual position ids, so the temporal
        # whole-block reuse kernel is intentionally bypassed.
        if self.refine_tokens:
            self.block_selection = False
        if center_bits not in {4, 8, 16}:
            raise ValueError("center_bits must be one of 4, 8, or 16")
        self.compact_metadata = bool(compact_metadata)
        self.center_bits = int(center_bits)
        if self.center_bits < 16 and not self.compact_metadata:
            raise ValueError("quantized centers require compact_metadata")
        if router_backend not in {"torch", "triton"}:
            raise ValueError("router_backend must be torch or triton")
        self.router_backend = router_backend
        self.streaming_component_reserve = int(os.environ.get(
            "SHADOWKV_STREAMING_COMPONENT_RESERVE", "64"
        ))
        if self.streaming_component_reserve < 0:
            raise ValueError(
                "SHADOWKV_STREAMING_COMPONENT_RESERVE must be non-negative"
            )
        super().__init__(config, **kwargs)
        self.refine_query_factor: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.refine_key_factor: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.refine_key_codes: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.refine_key_scale: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self._refine_prompt_query: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self._refine_prepare_layer = 0
        # Variable-order routing is the method, not a two-center special case.
        # Every positional block must expose the full nested 1..S path; the
        # allocator may still spend a small mean quota (for example 1.25), but
        # it is free to concentrate several upgrades on a difficult block.
        if max_components is None:
            max_components = block_size
        if int(max_components) != block_size:
            raise ValueError(
                "adaptive centroid routing requires the full 1..block_size "
                "component path; a 1/2-only cap is not supported"
            )
        self.router_slots = block_size
        if self.center_placement in {
            "qone", "qmass_one", "qtangent_one"
        } and (
            self.router_slots != block_size or not self.compact_metadata
        ):
            raise ValueError(
                "query-aware placement requires max_components=block_size and "
                "compact_metadata so the mean quota can remain small"
            )
        block_shape = (
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            self.max_blocks,
        )
        self.component_count = torch.zeros(
            block_shape, device=self.device, dtype=torch.uint8
        )
        if self.compact_metadata:
            # Each layer owns a CSR-like component pool.  Blocks store only a
            # uint8 length; packed_component_block is the segment id used by
            # the GPU scatter-logsumexp.  Capacity follows the global average
            # component budget, not the maximum per-block order.
            self.router_centroids: list[torch.Tensor | None] = [
                None for _ in range(self.num_layers)
            ]
            self.router_center_scale: list[torch.Tensor | None] = [
                None for _ in range(self.num_layers)
            ]
            self.router_log_counts: list[torch.Tensor | None] = [
                None for _ in range(self.num_layers)
            ]
            self.router_alpha: list[torch.Tensor | None] = [
                None for _ in range(self.num_layers)
            ]
            self.router_component_block: list[torch.Tensor | None] = [
                None for _ in range(self.num_layers)
            ]
            self.router_component_start: list[torch.Tensor | None] = [
                torch.full(
                    (self.batch_size, self.num_key_value_heads, self.max_blocks),
                    -1, device=self.device, dtype=torch.int32,
                )
                for _ in range(self.num_layers)
            ]
            self.router_component_used = [0 for _ in range(self.num_layers)]
        else:
            shape = (*block_shape, self.router_slots)
            self.router_centroids = torch.zeros(
                (*shape, self.head_dim), device=self.device, dtype=self.dtype
            )
            self.router_center_scale = None
            self.router_log_counts = torch.full(
                shape, float("-inf"), device=self.device, dtype=self.dtype
            )
            self.router_alpha = torch.zeros(
                shape, device=self.device, dtype=torch.float32
            )
            self.router_component_block = None
            self.router_component_start = None
        self.allocation_threshold = torch.full(
            (
                self.num_layers,
                self.batch_size,
                self.num_key_value_heads,
            ),
            float("nan"),
            device=self.device,
            dtype=torch.float32,
        )
        self.router_value_mean = torch.zeros(
            (*block_shape, self.head_dim),
            device=self.device, dtype=self.dtype,
        )
        if self.component_vmean_allocation:
            if self.compact_metadata:
                self.router_component_value_mean: list[torch.Tensor | None] = [
                    None for _ in range(self.num_layers)
                ]
            else:
                self.router_component_value_mean = torch.zeros(
                    (*block_shape, self.router_slots, self.head_dim),
                    device=self.device, dtype=self.dtype,
                )
        else:
            self.router_component_value_mean = None
        self._last_ragged_query: torch.Tensor | None = None
        self._last_ragged_block_logits: torch.Tensor | None = None
        self._last_ragged_component_logits: torch.Tensor | None = None
        self._last_ragged_candidate_fraction_groups: torch.Tensor | None = None
        self.incremental_fitter: IncrementalTwoCentroidGraph | None = None
        self._placement_query: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        # A sealed decode block remains exact in the recent window for several
        # steps.  Spread its unchanged metadata construction over the next
        # block_size steps instead of stalling every eighth token on all
        # layers at once.  The representation is complete well before the
        # block can enter the retrieval candidate range.
        self.defer_sealed_build = (
            os.environ.get("SHADOWKV_DEFER_SEALED_BUILD", "1") == "1"
            and self.recent_tokens >= self.block_size
        )
        self.deferred_block_keys = (
            torch.empty(
                self.num_layers,
                self.batch_size,
                self.num_key_value_heads,
                self.block_size,
                self.head_dim,
                device=self.compute_device,
                dtype=self.dtype,
            )
            if self.defer_sealed_build else None
        )
        self.deferred_block_id = [-1] * self.num_layers
        self.deferred_block_end = [-1] * self.num_layers

    def print_stats(self) -> None:
        super().print_stats()
        configured_mean = (
            "adaptive"
            if self.center_allocation in {
                "tail_relative_rate_distortion",
                "tail_absolute_rate_distortion",
                "tail_absolute_marginal",
                "tail_mass_rate_distortion",
                "tail_distortion_threshold",
                "tail_group_distortion_target",
                "tail_demand_adaptive_rate_distortion",
                "tail_hierarchical_log_rate_distortion",
                "tail_hierarchical_floor_rate_distortion",
            }
            else f"{1.0 + self.extra_fraction:g}"
        )
        refinement = (
            f"{self.refine_candidate_ratio:g} ratio"
            if self.refine_candidate_ratio is not None
            else f"{self.refine_factor:g}x"
        )
        print(
            "STREAMING_ADAPTIVE_CENTROID_LSE | exact KV backing | "
            f"mean components {configured_mean} | "
            f"max components {self.router_slots} | "
            f"placement {self.center_placement} | "
            f"query-mass bank {self.query_mass_bank_size}/"
            f"window {self.query_mass_window} "
            f"objective {self.query_mass_objective} "
            f"GQA {self.query_mass_gqa_reduce} | "
            f"allocation {self.center_allocation} | "
            f"relative/absolute-RD penalty {self.relative_rd_penalty:g}/"
            f"{self.absolute_rd_penalty:g} | "
            f"density {self.density_signal_mode}/{self.density_weight_mode} "
            f"shrink {self.density_shrinkage:g} "
            f"head-trust {self.density_head_trust_power:g} "
            f"value-power {self.density_value_power:g} | "
            f"cost {self.self_lse_cost} beta {self.self_lse_cost_beta:g} | "
            f"dispersion correction {self.center_dispersion_correction} | "
            f"router {self.router_backend} | "
            f"metadata {'compact' if self.compact_metadata else 'dense'} "
            f"center-int{self.center_bits} | "
            f"exact-LSE refinement {refinement} "
            f"{'tokens' if self.refine_tokens else 'blocks'} | "
            f"component candidates {self.component_candidate_refinement} | "
            f"matched-allocation {self.match_token_allocation} | "
            f"component-V allocation {self.component_vmean_allocation} | "
            f"component-V blend {self.component_vmean_blend:g} | "
            f"token-GQA {self.refine_token_gqa_reduce} | "
            f"token-GQA-UCB {self.refine_token_gqa_ucb:g} | "
            f"refine-sketch rank {self.refine_sketch_rank} "
            f"int{self.refine_sketch_bits} {self.refine_sketch_basis} | "
            f"temperatures {self.self_lse_temperatures} | "
            f"GQA {'query-mean-first' if self.query_group_mean else self.group_reduce}"
        )

    def _reset_metadata(self) -> None:
        if self.compact_metadata:
            for layer_idx in range(self.num_layers):
                if self.router_centroids[layer_idx] is not None:
                    self.router_centroids[layer_idx].zero_()
                    if self.router_center_scale[layer_idx] is not None:
                        self.router_center_scale[layer_idx].zero_()
                    self.router_log_counts[layer_idx].fill_(float("-inf"))
                    self.router_alpha[layer_idx].zero_()
                    self.router_component_block[layer_idx].zero_()
                    self.router_component_start[layer_idx].fill_(-1)
                self.router_component_used[layer_idx] = 0
        else:
            self.router_centroids.zero_()
            self.router_log_counts.fill_(float("-inf"))
            self.router_alpha.zero_()
        self.component_count.zero_()
        self.allocation_threshold.fill_(float("nan"))
        self.router_value_mean.zero_()
        if self.component_vmean_allocation:
            if self.compact_metadata:
                for value_mean in self.router_component_value_mean:
                    if value_mean is not None:
                        value_mean.zero_()
            else:
                self.router_component_value_mean.zero_()
        for layer_idx in range(self.num_layers):
            self._placement_query[layer_idx] = None
            self._refine_prompt_query[layer_idx] = None
            self.refine_query_factor[layer_idx] = None
            self.refine_key_factor[layer_idx] = None
            self.refine_key_codes[layer_idx] = None
            self.refine_key_scale[layer_idx] = None
        self._refine_prepare_layer = 0
        self.deferred_block_id[:] = [-1] * self.num_layers
        self.deferred_block_end[:] = [-1] * self.num_layers

    def _before_streaming_append(self, layer_idx: int) -> None:
        """Build one pending layer on its deterministic amortization slot."""
        block_id = self.deferred_block_id[layer_idx]
        if block_id < 0:
            return
        age = (
            self.block_state[layer_idx].total_tokens
            - self.deferred_block_end[layer_idx]
        )
        delay = layer_idx % self.block_size
        if age < delay:
            return
        keys = self.deferred_block_keys[layer_idx].unsqueeze(2)
        self.deferred_block_id[layer_idx] = -1
        self.deferred_block_end[layer_idx] = -1
        self._build_blocks(layer_idx, (block_id,), keys)

    def _build_or_defer_blocks(
        self,
        layer_idx: int,
        block_ids: tuple[int, ...],
        block_keys: torch.Tensor,
    ) -> None:
        """Defer post-prefill hierarchy fitting without changing its result.

        With a batch-flush frame the shared queue is strictly better than the
        per-layer stagger below: the stagger spreads one fit per layer over
        ``block_size`` tokens, so every token still pays a share of the whole
        36-layer bill, while the queue fits ``update_interval / block_size``
        blocks in one vectorised call once per flush -- the same thing
        ParisKV's ``dynamic_update_interval`` does, and the reason that
        interval exists.  The stagger stays for the degenerate frame, where
        there is no buffer to queue into.
        """
        if self.update_interval != self.block_size:
            super()._build_or_defer_blocks(layer_idx, block_ids, block_keys)
            return
        can_defer = (
            self.defer_sealed_build
            and len(block_ids) == 1
            and not torch.isnan(self.allocation_threshold[layer_idx]).all()
        )
        if not can_defer:
            self._build_blocks(layer_idx, block_ids, block_keys)
            return
        # The previous block must have been consumed within the preceding
        # block_size-token interval.  Fall back safely if an unusual batched
        # append violates that steady-state contract.
        if self.deferred_block_id[layer_idx] >= 0:
            self._before_streaming_append(layer_idx)
        if self.deferred_block_id[layer_idx] >= 0:
            self._build_blocks(layer_idx, block_ids, block_keys)
            return
        block_id = int(block_ids[0])
        delay = layer_idx % self.block_size
        if delay == 0:
            self._build_blocks(layer_idx, block_ids, block_keys)
            return
        self.deferred_block_keys[layer_idx].copy_(block_keys.squeeze(2))
        self.deferred_block_id[layer_idx] = block_id
        self.deferred_block_end[layer_idx] = (block_id + 1) * self.block_size

    def _fit_refine_sketch(
        self,
        layer_idx: int,
        keys: torch.Tensor,
    ) -> None:
        """Fit one deployable rank sketch from causal prompt Q/K moments."""
        if not self.refine_sketch_rank:
            return
        prompt_query = self._refine_prompt_query[layer_idx]
        if prompt_query is None:
            raise RuntimeError("refinement sketch did not observe prompt queries")
        query = prompt_query.float().view(
            self.batch_size,
            self.num_key_value_heads,
            self.num_key_value_groups,
            prompt_query.shape[-2],
            self.head_dim,
        )
        if self.query_group_mean:
            query = query.mean(2)
        else:
            query = query.permute(0, 1, 3, 2, 4).flatten(2, 3)
        usable = keys.shape[-2] // self.block_size * self.block_size
        n_blocks = usable // self.block_size
        take = min(512, n_blocks)
        block_index = torch.linspace(
            0, n_blocks - 1, take, device=keys.device
        ).round().long()
        offset = torch.arange(self.block_size, device=keys.device)
        position = (
            block_index[:, None] * self.block_size + offset[None]
        ).flatten()
        sampled_key = keys.index_select(-2, position).float()

        query_factor = []
        key_factor = []
        for head in range(self.num_key_value_heads):
            q = query[:, head].reshape(-1, self.head_dim)
            k = sampled_key[:, head].reshape(-1, self.head_dim)
            q_moment = q.T @ q / max(1, len(q))
            k_moment = k.T @ k / max(1, len(k))
            if self.refine_sketch_basis == "key_pca":
                q_factor, k_factor = _key_pca_factors(
                    k_moment, self.refine_sketch_rank
                )
            else:
                q_factor, k_factor = _optimal_bilinear_factors(
                    q_moment, k_moment, self.refine_sketch_rank
                )
            query_factor.append(q_factor)
            key_factor.append(k_factor)
        self.refine_query_factor[layer_idx] = torch.stack(query_factor)
        self.refine_key_factor[layer_idx] = torch.stack(key_factor)
        self._refine_prompt_query[layer_idx] = None

    def _encode_refine_blocks(
        self,
        layer_idx: int,
        ids: torch.Tensor,
        blocks: torch.Tensor,
    ) -> None:
        if not self.refine_sketch_rank:
            return
        factor = self.refine_key_factor[layer_idx]
        if factor is None:
            raise RuntimeError("refinement key factor has not been fitted")
        projected = torch.einsum(
            "bhnsd,hdr->bhnsr", blocks.float(), factor.float()
        )
        encoded, scale = _quantize_centers(
            projected, self.refine_sketch_bits
        )
        if self.refine_key_codes[layer_idx] is None:
            width = (
                self.refine_sketch_rank
                if self.refine_sketch_bits != 4
                else (self.refine_sketch_rank + 1) // 2
            )
            dtype = (
                self.dtype if self.refine_sketch_bits == 16 else
                torch.int8 if self.refine_sketch_bits == 8 else torch.uint8
            )
            self.refine_key_codes[layer_idx] = torch.zeros(
                self.batch_size,
                self.num_key_value_heads,
                self.max_blocks,
                self.block_size,
                width,
                device=self.compute_device,
                dtype=dtype,
            )
            if self.refine_sketch_bits != 16:
                self.refine_key_scale[layer_idx] = torch.zeros(
                    self.batch_size,
                    self.num_key_value_heads,
                    self.max_blocks,
                    self.block_size,
                    device=self.compute_device,
                    dtype=torch.float32,
                )
        self.refine_key_codes[layer_idx].index_copy_(2, ids, encoded)
        if scale is not None:
            self.refine_key_scale[layer_idx].index_copy_(2, ids, scale)

    def _refine_sketch_logits(
        self,
        layer_idx: int,
        query: torch.Tensor,
        candidate_blocks: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate token logits for coarse candidates without exact K."""
        stored = self.refine_key_codes[layer_idx]
        if stored is None:
            raise RuntimeError("refinement key codes have not been built")
        width = stored.shape[-1]
        index = candidate_blocks[..., None, None].expand(
            *candidate_blocks.shape, self.block_size, width
        )
        encoded = stored.gather(2, index)
        scale = self.refine_key_scale[layer_idx]
        candidate_scale = None
        if scale is not None:
            scale_index = candidate_blocks[..., None].expand(
                *candidate_blocks.shape, self.block_size
            )
            candidate_scale = scale.gather(2, scale_index)
        projected_key = _dequantize_centers(
            encoded,
            candidate_scale,
            self.refine_sketch_bits,
            self.refine_sketch_rank,
            torch.float32,
        )
        query_factor = self.refine_query_factor[layer_idx]
        if query_factor is None:
            raise RuntimeError("refinement query factor has not been fitted")
        projected_query = torch.einsum(
            "bhgqd,hdr->bhgqr", query.float(), query_factor.float()
        )
        return torch.einsum(
            "bhgqr,bhcsr->bhgqcs", projected_query, projected_key
        ) / math.sqrt(self.head_dim)

    def prefill_kv_cache(
        self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> None:
        self._fit_refine_sketch(layer_idx, key_states_roped)
        if (
            self.center_placement in {
                "qone", "qmass_one", "qmass_bank", "qtangent_one", "self_qanchor"
            }
            or self.center_allocation == "anchor_tail_cvar"
            or (
                self.center_placement == "robust_trimmed"
                and self.robust_query_source == "final"
            )
        ):
            if query is None:
                raise ValueError(
                    "query-aware placement requires the final prompt query"
                )
            # The caller passes a view of the final prompt position.  Clone it
            # so the cache cannot retain the full prefill-query backing tensor.
            self._placement_query[layer_idx] = query.detach().clone()
        super().prefill_kv_cache(
            new_v_cache, layer_idx, key_states_roped, query
        )

    def prepare_prefill_query(self, query: torch.Tensor) -> torch.Tensor:
        """Return the causal query evidence consumed by center placement.

        Existing query-aware controls retain their single final-prompt query.
        ``qmass_bank`` samples a fixed-size bank uniformly from the final
        prompt window.  These queries have already occurred at materialization
        time; future decode queries never enter the representation.
        """
        if self.refine_sketch_rank:
            if self._refine_prepare_layer >= self.num_layers:
                raise RuntimeError("too many prefill-query preparations")
            count = min(self.refine_sketch_prompt_samples, query.shape[-2])
            index = torch.linspace(
                0, query.shape[-2] - 1, count, device=query.device
            ).round().long()
            self._refine_prompt_query[self._refine_prepare_layer] = (
                query.index_select(-2, index).detach()
            )
            self._refine_prepare_layer += 1
        if self.center_placement != "qmass_bank":
            return query[..., -1:, :]
        start = max(0, query.shape[-2] - self.query_mass_window)
        count = min(self.query_mass_bank_size, query.shape[-2] - start)
        index = torch.linspace(
            start, query.shape[-2] - 1, count,
            device=query.device,
        ).round().long()
        return query.index_select(-2, index)

    def _initial_packed_capacity(self) -> int:
        average = 1.0 + self.extra_fraction
        expected = int(math.ceil(self.max_blocks * average))
        return min(
            self.max_blocks * self.router_slots,
            expected + max(self.streaming_component_reserve, self.router_slots),
        )

    def _allocate_packed_layer(self, layer_idx: int, capacity: int) -> None:
        width = self.head_dim if self.center_bits != 4 else (self.head_dim + 1) // 2
        data_dtype = self.dtype if self.center_bits == 16 else (
            torch.int8 if self.center_bits == 8 else torch.uint8
        )
        prefix = (self.batch_size, self.num_key_value_heads, capacity)
        self.router_centroids[layer_idx] = torch.zeros(
            (*prefix, width), device=self.device, dtype=data_dtype
        )
        self.router_center_scale[layer_idx] = (
            None if self.center_bits == 16 else torch.zeros(
                prefix, device=self.device, dtype=torch.float32
            )
        )
        self.router_log_counts[layer_idx] = torch.full(
            prefix, float("-inf"), device=self.device, dtype=self.dtype
        )
        self.router_alpha[layer_idx] = torch.zeros(
            prefix, device=self.device, dtype=torch.float32
        )
        self.router_component_block[layer_idx] = torch.zeros(
            prefix, device=self.device, dtype=torch.int32
        )
        if self.component_vmean_allocation:
            self.router_component_value_mean[layer_idx] = torch.zeros(
                (*prefix, self.head_dim), device=self.device, dtype=self.dtype
            )

    def _ensure_packed_capacity(self, layer_idx: int, needed: int) -> None:
        current = self.router_centroids[layer_idx]
        if current is None:
            self._allocate_packed_layer(
                layer_idx, max(needed, self._initial_packed_capacity())
            )
            return
        if needed <= current.shape[-2]:
            return
        old_capacity = current.shape[-2]
        new_capacity = max(needed, int(math.ceil(old_capacity * 1.5)))
        old = (
            self.router_centroids[layer_idx],
            self.router_center_scale[layer_idx],
            self.router_log_counts[layer_idx],
            self.router_alpha[layer_idx],
            self.router_component_block[layer_idx],
            None if not self.component_vmean_allocation else
            self.router_component_value_mean[layer_idx],
        )
        used = self.router_component_used[layer_idx]
        self._allocate_packed_layer(layer_idx, new_capacity)
        self.router_centroids[layer_idx][..., :used, :].copy_(old[0][..., :used, :])
        if old[1] is not None:
            self.router_center_scale[layer_idx][..., :used].copy_(old[1][..., :used])
        self.router_log_counts[layer_idx][..., :used].copy_(old[2][..., :used])
        self.router_alpha[layer_idx][..., :used].copy_(old[3][..., :used])
        self.router_component_block[layer_idx][..., :used].copy_(old[4][..., :used])
        if self.component_vmean_allocation:
            self.router_component_value_mean[layer_idx][..., :used, :].copy_(
                old[5][..., :used, :]
            )

    def _store_materialized_components(
        self,
        layer_idx: int,
        ids: torch.Tensor,
        centers: torch.Tensor,
        log_counts: torch.Tensor,
        alpha: torch.Tensor,
        counts: torch.Tensor,
        component_value_mean: torch.Tensor | None = None,
    ) -> None:
        """Store a newly sealed group in dense or compact representation."""
        if not self.center_dispersion_correction:
            alpha = torch.zeros_like(alpha)
        self.component_count[layer_idx].index_copy_(
            2, ids, counts.to(torch.uint8)
        )
        if not self.compact_metadata:
            self.router_centroids[layer_idx].index_copy_(2, ids, centers)
            self.router_log_counts[layer_idx].index_copy_(2, ids, log_counts)
            self.router_alpha[layer_idx].index_copy_(2, ids, alpha)
            if self.component_vmean_allocation:
                if component_value_mean is None:
                    raise RuntimeError("component value means are missing")
                self.router_component_value_mean[layer_idx].index_copy_(
                    2, ids, component_value_mean
                )
            return

        groups = self.batch_size * self.num_key_value_heads
        n_blocks, slots, dim = centers.shape[-3:]
        center_groups = centers.reshape(groups, n_blocks, slots, dim)
        log_groups = log_counts.reshape(groups, n_blocks, slots)
        alpha_groups = alpha.reshape(groups, n_blocks, slots)
        count_groups = counts.reshape(groups, n_blocks)
        value_groups = None
        if self.component_vmean_allocation:
            if component_value_mean is None:
                raise RuntimeError("component value means are missing")
            value_groups = component_value_mean.reshape(
                groups, n_blocks, slots, dim
            )
        active_per_group = count_groups.sum(-1)
        append = int(active_per_group.max().item())
        start = self.router_component_used[layer_idx]
        end = start + append
        self._ensure_packed_capacity(layer_idx, end)
        block_grid = ids[None, :, None].expand(groups, n_blocks, slots)

        stored_centers = self.router_centroids[layer_idx].reshape(
            groups, self.router_centroids[layer_idx].shape[-2], -1
        )
        stored_scale = self.router_center_scale[layer_idx]
        if stored_scale is not None:
            stored_scale = stored_scale.reshape(groups, -1)
        stored_log = self.router_log_counts[layer_idx].reshape(groups, -1)
        stored_alpha = self.router_alpha[layer_idx].reshape(groups, -1)
        stored_blocks = self.router_component_block[layer_idx].reshape(groups, -1)
        stored_starts = self.router_component_start[layer_idx].reshape(
            groups, self.max_blocks
        )
        stored_values = None
        if self.component_vmean_allocation:
            stored_values = self.router_component_value_mean[layer_idx].reshape(
                groups, self.router_component_value_mean[layer_idx].shape[-2], dim
            )

        # Fast path for an exact newly sealed decode block.  Every group
        # appends the same S singleton components, so doing eight per-head
        # Python iterations would launch dozens of tiny quantize/copy kernels
        # at every block boundary.  This batched form has identical storage.
        if (
            n_blocks == 1
            and append == slots
            and bool((count_groups == slots).all().item())
        ):
            exact_centers = center_groups[:, 0]
            encoded, scale = _quantize_centers(exact_centers, self.center_bits)
            stored_centers[:, start:end].copy_(encoded)
            if scale is not None:
                stored_scale[:, start:end].copy_(scale)
            stored_log[:, start:end].copy_(log_groups[:, 0])
            stored_alpha[:, start:end].copy_(alpha_groups[:, 0])
            stored_blocks[:, start:end].fill_(int(ids[0].item()))
            stored_starts[:, ids[0]].fill_(start)
            if stored_values is not None:
                stored_values[:, start:end].copy_(value_groups[:, 0])
            self.router_component_used[layer_idx] = end
            return

        arange_slots = torch.arange(slots, device=centers.device)
        for group in range(groups):
            valid = arange_slots[None, :] < count_groups[group, :, None]
            active_centers = center_groups[group][valid]
            active_log = log_groups[group][valid]
            active_alpha = alpha_groups[group][valid]
            active_blocks = block_grid[group][valid]
            active_values = None if value_groups is None else value_groups[group][valid]
            count = active_centers.shape[0]
            starts = start + torch.cat((
                count_groups.new_zeros(1), count_groups[group].cumsum(0)[:-1]
            ))
            stored_starts[group].index_copy_(0, ids, starts.to(torch.int32))
            encoded, scale = _quantize_centers(active_centers, self.center_bits)
            stored_centers[group, start : start + count].copy_(encoded)
            if scale is not None:
                stored_scale[group, start : start + count].copy_(scale)
            stored_log[group, start : start + count].copy_(active_log)
            stored_alpha[group, start : start + count].copy_(active_alpha)
            stored_blocks[group, start : start + count].copy_(active_blocks)
            if stored_values is not None:
                stored_values[group, start : start + count].copy_(active_values)
            # Unequal per-head append counts use inert padding so every head's
            # pool retains the same GEMM width without allocating r_max/block.
            if count < append:
                stored_log[group, start + count : end].fill_(float("-inf"))
                stored_alpha[group, start + count : end].zero_()
                stored_blocks[group, start + count : end].zero_()
        self.router_component_used[layer_idx] = end

    def H2D(self) -> None:
        super().H2D()
        if (
            self.router_backend == "triton"
            and self.router_slots == 2
            and self.block_size == 8
            and self.self_lse_temperatures == (1.0,)
            and self.compute_device.type == "cuda"
            and self.incremental_fitter is None
            and not self.component_vmean_allocation
            and self.center_placement == "self"
            and self.center_allocation == "self_lse"
        ):
            self.incremental_fitter = IncrementalTwoCentroidGraph(
                batch_size=self.batch_size,
                heads=self.num_key_value_heads,
                dim=self.head_dim,
                device=self.compute_device,
                dtype=self.dtype,
            )

    def _prefill_threshold(
        self, risk_one: torch.Tensor
    ) -> torch.Tensor:
        n_blocks = risk_one.shape[-1]
        n_extra = int(round(self.extra_fraction * n_blocks))
        if n_extra == 0:
            return torch.full_like(risk_one[..., 0], float("inf"))
        if n_extra == n_blocks:
            return torch.full_like(risk_one[..., 0], float("-inf"))
        return torch.topk(risk_one, k=n_extra, dim=-1).values[..., -1]

    def _prefill_multilevel_threshold(
        self, risk_path: torch.Tensor
    ) -> torch.Tensor:
        """Priority cutoff reproducing the prefill minimax allocation."""
        n_blocks, n_components = risk_path.shape[-2:]
        n_extra = int(round(self.extra_fraction * n_blocks))
        if n_extra == 0:
            return torch.full_like(risk_path[..., 0, 0], float("inf"))
        if n_extra == n_blocks * (n_components - 1):
            return torch.full_like(risk_path[..., 0, 0], float("-inf"))
        priorities = risk_path[..., :-1].reshape(
            *risk_path.shape[:-2], n_blocks * (n_components - 1)
        )
        return torch.topk(priorities, k=n_extra, dim=-1).values[..., -1]

    def _prefill_marginal_threshold(
        self, risk_path: torch.Tensor
    ) -> torch.Tensor:
        """Cutoff on concavified r->r+1 distortion reductions."""
        n_blocks, n_components = risk_path.shape[-2:]
        n_extra = int(round(self.extra_fraction * n_blocks))
        if n_extra == 0:
            return torch.full_like(risk_path[..., 0, 0], float("inf"))
        if n_extra == n_blocks * (n_components - 1):
            return torch.full_like(risk_path[..., 0, 0], float("-inf"))
        gain = (risk_path[..., :-1] - risk_path[..., 1:]).clamp_min(0)
        # Project onto diminishing marginal gains.  This is the lower
        # concave envelope needed for exact discrete water filling.
        gain = torch.cummin(gain, dim=-1).values
        priorities = gain.reshape(
            *risk_path.shape[:-2], n_blocks * (n_components - 1)
        )
        return torch.topk(priorities, k=n_extra, dim=-1).values[..., -1]

    def _placement_anchor_bank(
        self, layer_idx: int, blocks: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast the latest causal query as an attention-scaled proxy.

        Self-K supplies eight query-free directions per block.  This bank adds
        the final query of the available prefix without replacing those
        directions.  For GQA, query-mean routing contributes one anchor per KV
        head; otherwise all query heads remain separate anchors.
        """
        query = self._placement_query[layer_idx]
        if query is None:
            raise RuntimeError("self-qanchor placement query was not observed")
        if query.shape[1] != self.num_key_value_heads:
            query = query.view(
                self.batch_size,
                self.num_key_value_heads,
                self.num_key_value_groups,
                query.shape[-2],
                self.head_dim,
            )
            if self.query_group_mean:
                query = query.mean(dim=2)
            else:
                query = query.flatten(2, 3)
        if query.ndim != 4:
            raise ValueError("placement anchor bank must have shape [B,H,Q,D]")
        query = query.float() / math.sqrt(self.head_dim)
        return query[:, :, None].expand(
            -1, -1, blocks.shape[-3], -1, -1
        )

    def _build_blocks(
        self, layer_idx: int, block_ids: tuple[int, ...],
        block_keys: torch.Tensor | None = None,
    ) -> None:
        if not block_ids:
            return
        ids, blocks = self._load_block_keys(
            layer_idx, block_ids, block_keys
        )
        self._encode_refine_blocks(layer_idx, ids, blocks)

        threshold = self.allocation_threshold[layer_idx]
        initial = bool(torch.isnan(threshold).all())
        if initial:
            first, last = self.block_state[layer_idx].candidate_block_range
            candidate_mask = (ids >= first) & (ids < last)
            candidate_positions = candidate_mask.nonzero(as_tuple=False).flatten()
            if not candidate_positions.numel():
                # A short prompt can be covered completely by prefix/recent
                # exact regions.  Fit its sealed blocks now so their metadata
                # is ready when they age into the candidate interval; price
                # the initial hierarchy over all of them.
                candidate_positions = torch.arange(
                    ids.numel(), device=ids.device
                )

        block_values = None
        if self.store_router_value_mean:
            value_source = self.v_cache[layer_idx]
            value_ids = ids.to(value_source.device)
            offsets = torch.arange(
                self.block_size, device=value_source.device
            )
            value_positions = (
                value_ids[:, None] * self.block_size + offsets[None]
            ).reshape(-1)
            block_values = value_source.index_select(
                2, value_positions
            ).reshape(
                self.batch_size,
                self.num_key_value_heads,
                len(block_ids),
                self.block_size,
                self.head_dim,
            ).to(self.compute_device, non_blocking=self.offload)
            value_mean = block_values.float().mean(-2).to(dtype=self.dtype)
            self.router_value_mean[layer_idx].index_copy_(
                2, ids.to(self.compute_device), value_mean
            )

        if self.center_placement in {
            "qone", "qmass_one", "qmass_bank", "qtangent_one"
        }:
            placement_query = self._placement_query[layer_idx]
            if placement_query is None:
                raise RuntimeError(
                    "query-aware placement query has not been observed"
                )
            # ``qone`` deliberately treats the observed GQA query heads as a
            # small causal uncertainty set even when live routing later uses
            # their mean.  ``qmass_one`` instead matches mean-query routing
            # exactly, providing a clean semantic control.
            if (
                self.center_placement in {
                    "qmass_one", "qmass_bank", "qtangent_one"
                }
                and self.query_group_mean
                and (
                    self.center_placement == "qtangent_one"
                    or self.query_mass_gqa_reduce == "mean"
                )
                and placement_query.shape[1] != self.num_key_value_heads
            ):
                placement_query = placement_query.view(
                    self.batch_size,
                    self.num_key_value_heads,
                    self.num_key_value_groups,
                    placement_query.shape[-2],
                    self.head_dim,
                ).mean(dim=2)
            if self.center_placement == "qtangent_one":
                anchor = placement_query.float()
                if anchor.ndim == 4:
                    anchor = anchor[:, :, None]
                anchor_logits = torch.einsum(
                    "bhgqd,bhnsd->bhgqns", anchor, blocks.float()
                ) / math.sqrt(self.head_dim)
                anchor_weight = torch.softmax(anchor_logits, dim=-1)
                gradient = torch.einsum(
                    "bhgqns,bhnsd->bhgqnd",
                    anchor_weight,
                    blocks.float(),
                ).squeeze(2).squeeze(2)
                anchor_value = torch.logsumexp(
                    anchor_logits, dim=-1
                ).squeeze(2).squeeze(2)
                anchor_linear = torch.einsum(
                    "bhqd,bhnd->bhqn",
                    anchor.squeeze(2),
                    gradient,
                ).squeeze(2) / math.sqrt(self.head_dim)
                intercept = anchor_value - anchor_linear
                counts = torch.ones(
                    *blocks.shape[:-2],
                    device=blocks.device,
                    dtype=torch.long,
                )
                centers = torch.zeros(
                    *gradient.shape[:-1], self.router_slots, self.head_dim,
                    device=blocks.device, dtype=self.dtype,
                )
                log_counts = torch.full(
                    (*intercept.shape, self.router_slots),
                    -torch.inf,
                    device=blocks.device,
                    dtype=torch.float32,
                )
                alpha = torch.zeros_like(log_counts)
                centers[..., 0, :] = gradient.to(self.dtype)
                log_counts[..., 0] = intercept
                self._store_materialized_components(
                    layer_idx, ids, centers, log_counts, alpha, counts, None
                )
                if initial:
                    threshold.zero_()
                return
            if self.center_placement in {"qmass_one", "qmass_bank"}:
                selection_fraction = None
                if self.query_mass_objective in {
                    "topk_hinge", "topk_hinge_cvar"
                }:
                    selected_blocks = max(1, self.sparse_budget // self.block_size)
                    selection_fraction = min(
                        1.0, selected_blocks / max(1, blocks.shape[-3])
                    )
                query_risk, query_path, _ = fit_observed_query_mass_paths(
                    blocks,
                    placement_query,
                    objective=self.query_mass_objective,
                    tail_fraction=self.tail_cvar_fraction,
                    selection_fraction=selection_fraction,
                )
                counts = torch.ones_like(
                    query_risk[..., 0], dtype=torch.long
                )
                if initial:
                    local = candidate_positions
                    local_counts = allocate_concave_marginal_counts(
                        query_risk[..., local, :], self.extra_fraction
                    )
                    counts[..., local] = local_counts
                    threshold.zero_()
                centers, log_counts, alpha = _padded_adaptive_centroids(
                    blocks, query_path, counts, self.router_slots
                )
                self._store_materialized_components(
                    layer_idx, ids, centers, log_counts, alpha, counts, None
                )
                return
            # A self-K path supplies the finite baseline hypothesis; the one
            # causal query then chooses the best simple placement at every r.
            _, self_path = fit_agglomerative_self_lse_paths(
                blocks,
                temperatures=self.self_lse_temperatures,
                batch_blocks=self.self_lse_batch_blocks,
                cost_mode=self.self_lse_cost,
                cost_beta=self.self_lse_cost_beta,
            )
            query_risk, query_path, relevance = fit_observed_query_lse_paths(
                blocks, placement_query, self_path
            )
            counts = torch.ones_like(query_risk[..., 0], dtype=torch.long)
            if initial:
                local = candidate_positions
                local_counts = allocate_concave_marginal_counts(
                    query_risk[..., local, :],
                    self.extra_fraction,
                    relevance[..., local].sqrt(),
                )
                counts[..., local] = local_counts
                # A finite marker records that initial allocation is done.
                # Newly sealed blocks remain one-center while protected by the
                # exact recent window; a later campaign can study reallocation
                # separately without changing tonight's placement experiment.
                threshold.zero_()
            centers, log_counts, alpha = _padded_adaptive_centroids(
                blocks, query_path, counts, self.router_slots
            )
            component_value_mean = None
            if self.component_vmean_allocation:
                component_value_mean = _padded_component_value_means(
                    block_values, query_path, counts, self.router_slots
                )
            self._store_materialized_components(
                layer_idx, ids, centers, log_counts, alpha, counts,
                component_value_mean,
            )
            return

        if self.center_placement in {
            "angular", "residual_tail", "residual_self", "value_tail"
        }:
            if self.router_slots != 2:
                raise ValueError(
                    f"{self.center_placement} placement requires max_components=2"
                )
            if self.center_placement == "angular":
                risk, assignments = fit_angular_isolation_lse_paths(
                    blocks, temperatures=self.self_lse_temperatures
                )
                gain = (risk[..., 0] - risk[..., 1]).clamp_min(0)
            elif self.center_placement == "residual_tail":
                gain, assignments = fit_residual_tail_path(blocks)
            elif self.center_placement == "residual_self":
                risk, assignments = fit_residual_isolation_lse_paths(
                    blocks, temperatures=self.self_lse_temperatures
                )
                gain = (risk[..., 0] - risk[..., 1]).clamp_min(0)
            else:
                gain, assignments = fit_value_tail_path(blocks, block_values)
            counts = torch.ones_like(gain, dtype=torch.long)
            if initial:
                candidate_gain = gain[..., candidate_positions]
                threshold.copy_(self._prefill_threshold(candidate_gain))
                n_extra = int(round(
                    self.extra_fraction * candidate_positions.numel()
                ))
                if n_extra:
                    chosen_local = candidate_gain.topk(n_extra, dim=-1).indices
                    chosen = candidate_positions[chosen_local]
                    counts.scatter_(-1, chosen, 2)
                outside = ~candidate_mask
                if outside.any():
                    counts[..., outside] = 1 + (
                        gain[..., outside] >= threshold[..., None]
                    ).long()
            else:
                counts = 1 + (gain >= threshold[..., None]).long()

            centers, log_counts, alpha = _padded_adaptive_centroids(
                blocks, assignments, counts, self.router_slots
            )
            component_value_mean = None
            if self.component_vmean_allocation:
                component_value_mean = _padded_component_value_means(
                    block_values, assignments, counts, self.router_slots
                )
            self._store_materialized_components(
                layer_idx, ids, centers, log_counts, alpha, counts,
                component_value_mean,
            )
            return

        if (
            not initial
            and self.incremental_fitter is not None
            and len(block_ids) == 1
            and blocks.shape[-3] == 1
        ):
            centers, log_counts, alpha, counts, _ = (
                self.incremental_fitter.run(blocks, threshold)
            )
            self._store_materialized_components(
                layer_idx, ids, centers, log_counts, alpha, counts
            )
            return

        if self.router_slots == 2:
            solve_fraction = self.extra_fraction if initial else 1.0
            risk, assignments, counts = fit_lazy_exact_self_lse_allocation(
                blocks,
                extra_fraction=solve_fraction,
                temperatures=self.self_lse_temperatures,
            )
            allocation_priority = risk[..., 0]
            if self.center_allocation == "residual_radius":
                mean = blocks.float().mean(-2, keepdim=True)
                allocation_priority = (
                    blocks.float() - mean
                ).norm(dim=-1).amax(-1)
            if initial:
                candidate_risk = allocation_priority[..., candidate_positions]
                threshold.copy_(self._prefill_threshold(candidate_risk))

                # Spend the exact component quota only on blocks eligible for
                # retrieval. Prefix/recent blocks receive a provisional
                # representation for when they later enter the candidate pool.
                n_extra = int(round(
                    self.extra_fraction * candidate_positions.numel()
                ))
                counts = torch.ones_like(counts)
                if n_extra:
                    chosen_local = candidate_risk.topk(n_extra, dim=-1).indices
                    chosen = candidate_positions[chosen_local]
                    counts.scatter_(-1, chosen, 2)
                outside = ~candidate_mask
                if outside.any():
                    outside_counts = 1 + (
                        allocation_priority[..., outside] >= threshold[..., None]
                    ).long()
                    counts[..., outside] = outside_counts
            else:
                counts = 1 + (
                    allocation_priority >= threshold[..., None]
                ).long()
        else:
            anchor_proxy_queries = (
                self._placement_anchor_bank(layer_idx, blocks)
                if (
                    self.center_placement == "self_qanchor"
                    or self.center_allocation == "anchor_tail_cvar"
                ) else None
            )
            placement_proxy_queries = (
                anchor_proxy_queries
                if self.center_placement == "self_qanchor" else None
            )
            if self.center_placement == "robust_trimmed":
                if self.robust_query_source == "final":
                    placement_query = self._placement_query[layer_idx]
                    if placement_query is None:
                        raise RuntimeError(
                            "final robust proxy query has not been observed"
                        )
                    if (
                        self.query_group_mean
                        and placement_query.shape[1]
                        != self.num_key_value_heads
                    ):
                        placement_query = placement_query.view(
                            self.batch_size,
                            self.num_key_value_heads,
                            self.num_key_value_groups,
                            placement_query.shape[-2],
                            self.head_dim,
                        ).mean(dim=2)
                    risk, assignments, _ = (
                        fit_observed_query_trimmed_cvar_paths(
                            blocks,
                            placement_query,
                            placement_trim_fraction=self.robust_trim_fraction,
                            tail_fraction=self.tail_cvar_fraction,
                        )
                    )
                else:
                    _, assignments = fit_agglomerative_self_lse_paths(
                        blocks,
                        temperatures=self.self_lse_temperatures,
                        batch_blocks=self.self_lse_batch_blocks,
                        cost_mode="trimmed_gap",
                        cost_beta=self.robust_trim_fraction,
                    )
                    risk = evaluate_self_lse_cvar_path(
                        blocks,
                        assignments,
                        temperatures=self.self_lse_temperatures,
                        tail_fraction=self.tail_cvar_fraction,
                    )
            elif self.center_placement in {
                "residual_path", "angular_path", "norm_path"
            }:
                selection_mode = self.center_placement.removesuffix("_path")
                risk, assignments = fit_residual_tail_lse_paths(
                    blocks,
                    temperatures=self.self_lse_temperatures,
                    cost_mode=self.self_lse_cost,
                    cost_beta=self.self_lse_cost_beta,
                    selection_mode=selection_mode,
                )
            else:
                risk, assignments = fit_agglomerative_self_lse_paths(
                    blocks,
                    temperatures=self.self_lse_temperatures,
                    batch_blocks=self.self_lse_batch_blocks,
                    cost_mode=self.self_lse_cost,
                    cost_beta=self.self_lse_cost_beta,
                    extra_proxy_queries=placement_proxy_queries,
                )
            # Placement and rate allocation are distinct decisions.  A
            # tail-CVaR allocator can price any nested path; robust-trimmed
            # was merely the first placement for which it was implemented.
            # Re-evaluate the chosen non-robust path with the common tail
            # objective so the comparison changes placement only.
            if (
                self.center_allocation in {
                    "tail_cvar", "anchor_tail_cvar", "density_tail_cvar",
                    "tail_rate_distortion", "tail_relative_rate_distortion",
                    "tail_absolute_rate_distortion", "tail_absolute_marginal",
                    "tail_mass_rate_distortion",
                    "tail_distortion_threshold", "tail_group_distortion_target",
                    "tail_demand_adaptive_rate_distortion",
                    "tail_hierarchical_log_rate_distortion",
                    "tail_hierarchical_floor_rate_distortion",
                }
                and self.center_placement != "robust_trimmed"
            ):
                risk = evaluate_self_lse_cvar_path(
                    blocks,
                    assignments,
                    temperatures=self.self_lse_temperatures,
                    tail_fraction=self.tail_cvar_fraction,
                )
            risk = risk[..., : self.router_slots]
            assignments = assignments[..., : self.router_slots, :]
            allocation_path = risk
            if self.center_allocation == "anchor_tail_cvar":
                if anchor_proxy_queries is None:
                    raise RuntimeError("anchor-tail allocation has no query anchor")
                anchor_risk = evaluate_proxy_lse_gap_path(
                    blocks, assignments, anchor_proxy_queries
                )[..., : self.router_slots]
                allocation_path = (
                    (1.0 - self.qanchor_weight) * risk
                    + self.qanchor_weight * anchor_risk
                )
            if self.center_allocation in {
                "angular_marginal", "exposure_angular_marginal"
            }:
                allocation_path = evaluate_angular_radius_path(
                    blocks, assignments
                )
            density_weight: torch.Tensor | None = None
            if self.center_allocation == "exposure_angular_marginal":
                if initial:
                    density_weight = estimate_self_k_block_exposure(
                        blocks,
                        reference_count=self.exposure_reference_count,
                        temperature=self.self_lse_temperatures[0],
                    )
                else:
                    # Newly sealed decode blocks remain in the exact recent
                    # window.  A single-block call cannot estimate global
                    # exposure, so its eventual index entry uses neutral
                    # weight rather than a fabricated local probability.
                    density_weight = torch.ones_like(risk[..., 0])
            if self.center_allocation == "density_tail_cvar":
                density_weight = torch.ones_like(risk[..., 0])
                if initial:
                    candidate_blocks = blocks[..., candidate_positions, :, :]
                    raw_signal = _macro_angular_kde_maxmed_signal(
                        candidate_blocks,
                        page_blocks=(
                            self.density_page_tokens // self.block_size
                        ),
                        reference_count=self.density_reference_count,
                        kappa=self.density_kappa,
                        signal_mode=self.density_signal_mode,
                    )
                    if self.density_value_power:
                        candidate_block_values = block_values[
                            ..., candidate_positions, :, :
                        ]
                        # Exposure estimates how likely a block is to matter;
                        # its within-block V radius estimates the consequence
                        # of routing it through an imperfect K summary.  Their
                        # product is a query-free proxy for expected output
                        # distortion, rather than retained attention alone.
                        value_residual = (
                            candidate_block_values.float()
                            - candidate_block_values.float().mean(
                                dim=-2, keepdim=True
                            )
                        ).norm(dim=-1)
                        value_scale = torch.quantile(
                            value_residual, 0.90, dim=-1
                        ).clamp_min(1e-12)
                        value_scale = value_scale / value_scale.mean(
                            dim=-1, keepdim=True
                        ).clamp_min(1e-12)
                        raw_signal = raw_signal * value_scale.pow(
                            self.density_value_power
                        )
                    normalized_signal = raw_signal / raw_signal.mean(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-12)
                    candidate_weight = _head_trusted_density_weight(
                        normalized_signal,
                        risk[..., candidate_positions, :],
                        self.density_shrinkage,
                        self.density_head_trust_power,
                        up_only=self.density_weight_mode == "up_only",
                    )
                    density_weight[..., candidate_positions] = candidate_weight
            if self.center_allocation in {
                "residual_first", "residual_gate", "residual_weighted",
                "residual_marginal", "residual_rank_marginal",
            }:
                # Query-free cross-block allocation.  ``addressability`` is
                # the largest distance from the block mean: a key that the
                # one-body summary dilutes badly produces a large value.  The
                # placement path then determines how much of that difficulty
                # remains after each additional component.
                mean = blocks.float().mean(-2, keepdim=True)
                addressability = (
                    blocks.float() - mean
                ).square().sum(-1).amax(-1).sqrt()
                if self.center_allocation == "residual_first":
                    # Exact form of the strongest offline screen: spend the
                    # scarce x<=1 quota on the highest-residual blocks before
                    # giving any block a third component.
                    allocation_path = torch.cat((
                        addressability[..., None],
                        torch.zeros_like(risk[..., 1:]),
                    ), dim=-1)
                elif self.center_allocation == "residual_gate":
                    # The first priority is exactly the intrinsic residual
                    # score screened offline.  Later priorities decay by the
                    # fraction of self-LSE risk still unresolved, yielding a
                    # nested 1..S allocation rather than a hard 1/2 switch.
                    relative = risk / risk[..., :1].clamp_min(1e-12)
                    allocation_path = addressability[..., None] * relative
                elif self.center_allocation == "residual_weighted":
                    # Weighted minimax: minimize max_G A_G R_G(r_G).
                    allocation_path = addressability[..., None] * risk
                else:
                    # Weighted marginal water filling: minimize the separable
                    # concave surrogate sum_G A_G R_G(r_G).
                    gain = (risk[..., :-1] - risk[..., 1:]).clamp_min(0)
                    gain = torch.cummin(gain, dim=-1).values
                    if self.center_allocation == "residual_rank_marginal":
                        # Percentile normalization prevents a few norm-scale
                        # outliers from consuming several upgrades.  It is
                        # invariant to every monotone rescaling of A_G.
                        order = addressability.argsort(-1).argsort(-1)
                        addressability = (
                            order.to(addressability.dtype) + 1
                        ) / addressability.shape[-1]
                    allocation_path = torch.cat((
                        addressability[..., None] * gain,
                        torch.zeros_like(risk[..., :1]),
                    ), dim=-1)
            if initial:
                candidate_risk = risk[..., candidate_positions, :]
                candidate_allocation = allocation_path[
                    ..., candidate_positions, :
                ]
                if self.center_allocation in {
                    "rate_distortion", "tail_rate_distortion"
                }:
                    candidate_counts, penalty = (
                        allocate_rate_distortion_counts(
                            candidate_risk, self.extra_fraction
                        )
                    )
                    threshold.copy_(penalty)
                    counts = rate_distortion_counts(risk, penalty)
                elif self.center_allocation == "tail_relative_rate_distortion":
                    penalty = torch.full_like(
                        candidate_risk[..., 0, 0], self.relative_rd_penalty
                    )
                    candidate_counts = relative_rate_distortion_counts(
                        candidate_risk, penalty
                    )
                    threshold.copy_(penalty)
                    counts = relative_rate_distortion_counts(risk, penalty)
                elif self.center_allocation == "tail_absolute_rate_distortion":
                    # A fixed price in distortion units deliberately keeps
                    # absolute cross-prompt difficulty.  There is no global
                    # quota: each block independently chooses the order that
                    # minimizes tail risk plus the price of extra centers.
                    penalty = torch.full_like(
                        candidate_risk[..., 0, 0], self.absolute_rd_penalty
                    )
                    candidate_counts = rate_distortion_counts(
                        candidate_risk, penalty
                    )
                    threshold.copy_(penalty)
                    counts = rate_distortion_counts(risk, penalty)
                elif self.center_allocation == "tail_absolute_marginal":
                    penalty = torch.full_like(
                        candidate_risk[..., 0, 0], self.absolute_rd_penalty
                    )
                    candidate_counts = priced_concave_marginal_counts(
                        candidate_risk, penalty
                    )
                    threshold.copy_(penalty)
                    counts = priced_concave_marginal_counts(risk, penalty)
                elif self.center_allocation == "tail_mass_rate_distortion":
                    penalty = torch.full_like(
                        candidate_risk[..., 0, 0], self.absolute_rd_penalty
                    )
                    # A log-mass error R means a multiplicative mass error of
                    # exp(R)-1.  Pricing that quantity amplifies genuinely
                    # unresolved blocks/tasks without a task label or quota.
                    mass_risk = torch.expm1(risk.clamp_max(20.0))
                    candidate_mass_risk = mass_risk[
                        ..., candidate_positions, :
                    ]
                    candidate_counts = rate_distortion_counts(
                        candidate_mass_risk, penalty
                    )
                    threshold.copy_(penalty)
                    counts = rate_distortion_counts(mass_risk, penalty)
                elif self.center_allocation == "tail_distortion_threshold":
                    tolerance = torch.full_like(
                        candidate_risk[..., 0, 0], self.distortion_tolerance
                    )
                    candidate_counts = distortion_threshold_counts(
                        candidate_risk, tolerance
                    )
                    threshold.copy_(tolerance)
                    counts = distortion_threshold_counts(risk, tolerance)
                elif self.center_allocation == "tail_group_distortion_target":
                    target = torch.full_like(
                        candidate_risk[..., 0, 0],
                        self.group_distortion_target,
                    )
                    candidate_counts, penalty = (
                        allocate_distortion_target_counts(
                            candidate_risk, target
                        )
                    )
                    threshold.copy_(penalty)
                    counts = rate_distortion_counts(risk, penalty)
                elif self.center_allocation == "tail_demand_adaptive_rate_distortion":
                    candidate_counts, penalty, _ = (
                        demand_adaptive_rate_distortion_counts(
                            candidate_risk,
                            self.demand_low_penalty,
                            self.demand_high_penalty,
                            self.demand_threshold,
                        )
                    )
                    threshold.copy_(penalty)
                    counts = rate_distortion_counts(risk, penalty)
                elif self.center_allocation == "tail_hierarchical_log_rate_distortion":
                    candidate_counts, penalty, _ = (
                        hierarchical_log_rate_distortion_counts(
                            candidate_risk,
                            self.hierarchical_rd_alpha,
                            self.hierarchical_rd_beta,
                            self.hierarchical_rd_group,
                        )
                    )
                    threshold.copy_(penalty)
                    counts = rate_distortion_counts(risk, penalty)
                elif self.center_allocation == "tail_hierarchical_floor_rate_distortion":
                    candidate_counts, penalty, _ = (
                        hierarchical_log_rate_distortion_counts(
                            candidate_risk,
                            self.hierarchical_rd_alpha,
                            self.hierarchical_rd_beta,
                            self.hierarchical_rd_group,
                            base_penalty=self.hierarchical_rd_base,
                        )
                    )
                    threshold.copy_(penalty)
                    counts = rate_distortion_counts(risk, penalty)
                elif self.center_allocation in {
                    "marginal", "tail_cvar", "anchor_tail_cvar", "density_tail_cvar",
                    "angular_marginal", "exposure_angular_marginal",
                }:
                    candidate_weight = (
                        None if density_weight is None
                        else density_weight[..., candidate_positions]
                    )
                    candidate_counts = allocate_concave_marginal_counts(
                        candidate_allocation,
                        self.extra_fraction,
                        weight=candidate_weight,
                    )
                    gain = (
                        allocation_path[..., :-1]
                        - allocation_path[..., 1:]
                    ).clamp_min(0)
                    if density_weight is not None:
                        gain = gain * density_weight[..., None]
                    gain = torch.cummin(gain, dim=-1).values
                    candidate_gain = gain[..., candidate_positions, :].flatten(-2)
                    n_extra = int(round(
                        self.extra_fraction * candidate_positions.numel()
                    ))
                    if n_extra:
                        threshold.copy_(
                            candidate_gain.topk(n_extra, dim=-1).values[..., -1]
                        )
                    else:
                        threshold.fill_(float("inf"))
                    counts = 1 + (
                        gain >= threshold[..., None, None]
                    ).sum(-1)
                elif self.center_allocation == "self_lse":
                    candidate_counts = allocate_minimax_centroid_counts(
                        candidate_risk, self.extra_fraction
                    )
                    threshold.copy_(
                        self._prefill_multilevel_threshold(candidate_risk)
                    )
                    counts = 1 + (
                        risk[..., :-1] >= threshold[..., None, None]
                    ).sum(-1)
                else:
                    candidate_counts = allocate_minimax_centroid_counts(
                        candidate_allocation, self.extra_fraction
                    )
                    threshold.copy_(
                        self._prefill_multilevel_threshold(
                            candidate_allocation
                        )
                    )
                    counts = 1 + (
                        allocation_path[..., :-1]
                        >= threshold[..., None, None]
                    ).sum(-1)
                # Several vectorized allocation helpers intentionally execute
                # under inference mode. Clone before patching candidate blocks
                # so this path also works when the caller is not itself inside
                # a global inference-mode context (e.g. unit tests/tools).
                counts = counts.clone()
                counts[..., candidate_positions] = candidate_counts
            else:
                if self.center_allocation in {
                    "rate_distortion", "tail_rate_distortion"
                }:
                    counts = rate_distortion_counts(risk, threshold)
                elif self.center_allocation == "tail_relative_rate_distortion":
                    counts = relative_rate_distortion_counts(risk, threshold)
                elif self.center_allocation == "tail_absolute_rate_distortion":
                    counts = rate_distortion_counts(risk, threshold)
                elif self.center_allocation == "tail_absolute_marginal":
                    counts = priced_concave_marginal_counts(risk, threshold)
                elif self.center_allocation == "tail_mass_rate_distortion":
                    counts = rate_distortion_counts(
                        torch.expm1(risk.clamp_max(20.0)), threshold
                    )
                elif self.center_allocation == "tail_distortion_threshold":
                    counts = distortion_threshold_counts(risk, threshold)
                elif self.center_allocation == "tail_group_distortion_target":
                    counts = rate_distortion_counts(risk, threshold)
                elif self.center_allocation == "tail_demand_adaptive_rate_distortion":
                    counts = rate_distortion_counts(risk, threshold)
                elif self.center_allocation == "tail_hierarchical_log_rate_distortion":
                    counts = rate_distortion_counts(risk, threshold)
                elif self.center_allocation == "tail_hierarchical_floor_rate_distortion":
                    counts = rate_distortion_counts(risk, threshold)
                elif self.center_allocation in {
                    "marginal", "tail_cvar", "anchor_tail_cvar", "density_tail_cvar",
                    "angular_marginal", "exposure_angular_marginal",
                }:
                    gain = (
                        allocation_path[..., :-1]
                        - allocation_path[..., 1:]
                    ).clamp_min(0)
                    if density_weight is not None:
                        gain = gain * density_weight[..., None]
                    gain = torch.cummin(gain, dim=-1).values
                    counts = 1 + (
                        gain >= threshold[..., None, None]
                    ).sum(-1)
                elif self.center_allocation == "self_lse":
                    counts = 1 + (
                        risk[..., :-1] >= threshold[..., None, None]
                    ).sum(-1)
                else:
                    counts = 1 + (
                        allocation_path[..., :-1]
                        >= threshold[..., None, None]
                    ).sum(-1)

        centers, log_counts, alpha = _padded_adaptive_centroids(
            blocks, assignments, counts, self.router_slots
        )
        if self.tail_gap_correction_scale:
            if self.center_allocation not in {
                "tail_cvar", "anchor_tail_cvar", "density_tail_cvar",
                "tail_rate_distortion", "tail_relative_rate_distortion",
                "tail_absolute_rate_distortion", "tail_absolute_marginal",
                "tail_mass_rate_distortion",
                "tail_distortion_threshold", "tail_group_distortion_target",
                "tail_demand_adaptive_rate_distortion",
                "tail_hierarchical_log_rate_distortion",
                "tail_hierarchical_floor_rate_distortion",
            }:
                raise ValueError(
                    "tail-gap correction requires a tail-CVaR allocation"
                )
            # Equalize variable-resolution summaries by estimating the
            # Jensen log-mass still unresolved at the selected order.  The
            # same block scalar is folded into every valid component's
            # log-count, so this adds no metadata or decode-time operation.
            selected_gap = risk.gather(
                -1, (counts - 1).unsqueeze(-1)
            ).squeeze(-1)
            log_counts = log_counts + (
                self.tail_gap_correction_scale * selected_gap
            ).unsqueeze(-1)
        component_value_mean = None
        if self.component_vmean_allocation:
            component_value_mean = _padded_component_value_means(
                block_values,
                assignments,
                counts,
                self.router_slots,
            )
        self._store_materialized_components(
            layer_idx, ids, centers, log_counts, alpha, counts,
            component_value_mean,
        )

    def _packed_block_logits(
        self,
        layer_idx: int,
        query: torch.Tensor,
        first_block: int,
        last_block: int,
    ) -> torch.Tensor:
        used = self.router_component_used[layer_idx]
        if not used:
            raise RuntimeError("compact router metadata has not been built")
        stored = self.router_centroids[layer_idx][..., :used, :]
        scale = self.router_center_scale[layer_idx]
        if scale is not None:
            scale = scale[..., :used]
        centers = _dequantize_centers(
            stored, scale, self.center_bits, self.head_dim, query.dtype
        )
        log_counts = self.router_log_counts[layer_idx][..., :used].float()
        alpha = self.router_alpha[layer_idx][..., :used].float()
        component_block = self.router_component_block[layer_idx][
            ..., :used
        ].long()
        valid = (
            torch.isfinite(log_counts)
            & (component_block >= first_block)
            & (component_block < last_block)
        )
        component_logits = torch.einsum(
            "bhgqd,bhmd->bhgqm", query, centers
        ).float() / math.sqrt(self.head_dim)
        component_logits = component_logits + log_counts[:, :, None, None]
        query_norm2 = query.float().square().sum(-1) / self.head_dim
        component_logits = component_logits + query_norm2[..., None] * (
            alpha[:, :, None, None]
        )
        component_logits = component_logits.masked_fill(
            ~valid[:, :, None, None], float("-inf")
        )
        n_blocks = last_block - first_block
        relative = (component_block - first_block).clamp(0, n_blocks - 1)
        index = relative[:, :, None, None].expand_as(component_logits)
        output_shape = (*component_logits.shape[:-1], n_blocks)
        block_max = torch.full(
            output_shape,
            float("-inf"),
            device=component_logits.device,
            dtype=component_logits.dtype,
        )
        block_max.scatter_reduce_(
            dim=-1,
            index=index,
            src=component_logits,
            reduce="amax",
            include_self=True,
        )
        gathered_max = block_max.gather(-1, index)
        centered = torch.where(
            valid[:, :, None, None],
            component_logits - gathered_max,
            torch.full_like(component_logits, float("-inf")),
        )
        block_sum = torch.zeros_like(block_max)
        block_sum.scatter_add_(dim=-1, index=index, src=centered.exp())
        return block_max + block_sum.log()

    def _block_logits(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        first_block: int,
        last_block: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = query_states.view(
            self.batch_size,
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.incoming_q_len,
            self.head_dim,
        )
        if self.query_group_mean:
            # ParisKV forms one retrieval query per KV head before scoring.
            # Keep singleton group/window axes so the remaining LSE pipeline
            # is identical to the ordinary per-query-head path.
            query = query.mean(dim=(2, 3), keepdim=True)
        if self.compact_metadata:
            self._last_ragged_component_logits = None
            if (
                self.router_backend == "triton"
                and self.incoming_q_len == 1
            ):
                logits = packed_block_logits(
                    query,
                    self.router_centroids[layer_idx],
                    self.router_center_scale[layer_idx],
                    self.router_log_counts[layer_idx],
                    self.router_alpha[layer_idx],
                    self.router_component_start[layer_idx],
                    self.component_count[layer_idx],
                    first_block=first_block,
                    last_block=last_block,
                    center_bits=self.center_bits,
                    max_components=self.router_slots,
                )
                return query, logits.unsqueeze(-2)
            return query, self._packed_block_logits(
                layer_idx, query, first_block, last_block
            )
        centers = self.router_centroids[layer_idx, :, :, first_block:last_block]
        log_counts = self.router_log_counts[layer_idx, :, :, first_block:last_block]
        alpha = self.router_alpha[layer_idx, :, :, first_block:last_block]
        use_triton = (
            self.router_backend == "triton"
            and self.router_slots == 2
            and self.incoming_q_len == 1
            and not self.query_group_mean
        )
        if use_triton:
            self._last_ragged_component_logits = None
            block_logits = two_slot_block_logits(
                query, centers, log_counts, alpha
            ).unsqueeze(-2)
        else:
            component_logits = torch.einsum(
                "bhgqd,bhcrd->bhgqcr", query, centers
            ).float() / math.sqrt(self.head_dim)
            component_logits = component_logits + log_counts.float()[
                :, :, None, None
            ]
            query_norm2 = query.float().square().sum(-1) / self.head_dim
            component_logits = component_logits + query_norm2[..., None, None] * (
                alpha[:, :, None, None]
            )
            self._last_ragged_component_logits = component_logits
            block_logits = torch.logsumexp(component_logits, dim=-1)
        return query, block_logits

    def _reduce_component_logits(
        self, component_logits: torch.Tensor
    ) -> torch.Tensor:
        """Apply temporal/GQA reduction before collapsing components."""
        shape = component_logits.shape
        probability = nn.functional.softmax(
            component_logits.flatten(-2), dim=-1
        ).view(shape)
        score = (
            probability.squeeze(-3)
            if self.incoming_q_len == 1
            else probability.sum(dim=-3)
        )
        if self.query_group_mean:
            score = score.squeeze(2)
        elif self.num_key_value_groups > 1:
            if self.group_reduce == "max":
                score = score.amax(dim=2)
            else:
                score = score.sum(dim=2)
        else:
            score = score.squeeze(2)
        return score

    def _reduce_block_logits(self, block_logits: torch.Tensor) -> torch.Tensor:
        probability = nn.functional.softmax(block_logits, dim=-1)
        score = (
            probability.squeeze(-2)
            if self.incoming_q_len == 1
            else probability.sum(dim=-2)
        )
        if self.query_group_mean:
            score = score.squeeze(2)
        elif self.num_key_value_groups > 1:
            if self.group_reduce == "max":
                score = score.amax(dim=2)
            else:
                score = score.sum(dim=2)
        else:
            score = score.squeeze(2)
        return score

    def _component_value_block_output(
        self,
        layer_idx: int,
        query: torch.Tensor,
        log_normalizer: torch.Tensor,
        first_block: int,
        last_block: int,
    ) -> torch.Tensor:
        """Approximate each block output with its fitted joint K/V mixture."""
        if not self.component_vmean_allocation:
            raise RuntimeError("component-V allocation is disabled")
        n_blocks = last_block - first_block
        if self.compact_metadata:
            used = self.router_component_used[layer_idx]
            centers = _dequantize_centers(
                self.router_centroids[layer_idx][..., :used, :],
                None if self.router_center_scale[layer_idx] is None else
                self.router_center_scale[layer_idx][..., :used],
                self.center_bits,
                self.head_dim,
                query.dtype,
            )
            log_counts = self.router_log_counts[layer_idx][..., :used].float()
            alpha = self.router_alpha[layer_idx][..., :used].float()
            component_block = self.router_component_block[layer_idx][
                ..., :used
            ].long()
            valid = (
                torch.isfinite(log_counts)
                & (component_block >= first_block)
                & (component_block < last_block)
            )
            logits = torch.einsum(
                "bhgqd,bhmd->bhgqm", query, centers
            ).float() / math.sqrt(self.head_dim)
            logits = logits + log_counts[:, :, None, None]
            query_norm2 = query.float().square().sum(-1) / self.head_dim
            logits = logits + query_norm2[..., None] * alpha[
                :, :, None, None
            ]
            logits = logits.masked_fill(
                ~valid[:, :, None, None], float("-inf")
            )
            probability = torch.exp(logits - log_normalizer)
            values = self.router_component_value_mean[layer_idx][
                ..., :used, :
            ].float()
            contribution = probability[..., None] * values[
                :, :, None, None, :, :
            ]
            relative = (component_block - first_block).clamp(
                0, n_blocks - 1
            )
            index = relative[:, :, None, None, :, None].expand_as(contribution)
            output = torch.zeros(
                (*contribution.shape[:-2], n_blocks, self.head_dim),
                device=contribution.device,
                dtype=contribution.dtype,
            )
            output.scatter_add_(-2, index, contribution)
            return output

        centers = self.router_centroids[
            layer_idx, :, :, first_block:last_block
        ]
        log_counts = self.router_log_counts[
            layer_idx, :, :, first_block:last_block
        ]
        alpha = self.router_alpha[
            layer_idx, :, :, first_block:last_block
        ]
        logits = torch.einsum(
            "bhgqd,bhnrd->bhgqnr", query, centers
        ).float() / math.sqrt(self.head_dim)
        logits = logits + log_counts.float()[:, :, None, None]
        query_norm2 = query.float().square().sum(-1) / self.head_dim
        logits = logits + query_norm2[..., None, None] * alpha[
            :, :, None, None
        ]
        probability = torch.exp(logits - log_normalizer[..., None])
        values = self.router_component_value_mean[
            layer_idx, :, :, first_block:last_block
        ].float()
        return torch.einsum(
            "bhgqnr,bhnrd->bhgqnd", probability, values
        )

    def _score_blocks(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        first_block: int,
        last_block: int,
    ) -> torch.Tensor:
        if (
            self.center_placement in {
                "qone", "qmass_one", "qtangent_one", "self_qanchor"
            }
            or self.center_allocation == "anchor_tail_cvar"
            or (
                self.center_placement == "robust_trimmed"
                and self.robust_query_source == "final"
            )
        ):
            self._placement_query[layer_idx] = query_states[
                ..., -1:, :
            ].detach().clone()
        _, block_logits = self._block_logits(
            layer_idx, query_states, first_block, last_block
        )
        return self._reduce_block_logits(block_logits)

    def _set_output_vmean_loss_for_ranking(
        self,
        layer_idx: int,
        query: torch.Tensor,
        block_logits: torch.Tensor,
        score: torch.Tensor,
        first_block: int,
        last_block: int,
    ) -> None:
        """Rebuild L_h(b) when a refinement changes the block ranking.

        The exact DP is only meaningful when its loss curve follows the same
        prefix order that will actually be gathered.  Previously block
        refinement changed the selector after the coarse loss was built.
        """
        if not self.head_alloc_mode.startswith("output_vmean"):
            return
        exact_positions = self._exact_position_ids(layer_idx)
        source = self.k_cache[layer_idx]
        exact_keys = source.index_select(
            2, exact_positions.to(source.device)
        ).to(self.compute_device, non_blocking=self.offload).float()
        fixed_logits = torch.einsum(
            "bhgqd,bhtd->bhgqt", query.float(), exact_keys
        ) / math.sqrt(self.head_dim)
        fixed_lse = torch.logsumexp(fixed_logits, dim=-1)
        candidate_lse = torch.logsumexp(block_logits.float(), dim=-1)
        joint_lse = torch.logaddexp(fixed_lse, candidate_lse)[..., None]
        fixed_probability = torch.exp(fixed_logits - joint_lse)
        block_probability = torch.exp(block_logits.float() - joint_lse)
        fixed_mass = fixed_probability.sum(-1)
        value_source = self.v_cache[layer_idx]
        exact_values = value_source.index_select(
            2, exact_positions.to(value_source.device)
        ).to(self.compute_device, non_blocking=self.offload).float()
        fixed_output = torch.einsum(
            "bhgqt,bhtd->bhgqd", fixed_probability, exact_values
        )
        block_value = self.router_value_mean[
            layer_idx, :, :, first_block:last_block
        ].float()
        block_output = block_probability[..., None] * block_value[
            :, :, None, None
        ]
        predicted_full = fixed_output + block_output.sum(-2)
        maximum = min(
            block_logits.shape[-1],
            self.head_alloc_max_budget // self.block_size,
        )
        loss = torch.full(
            (self.batch_size, self.num_key_value_heads, maximum + 1),
            float("inf"), device=self.compute_device,
        )
        order = score.argsort(-1, descending=True)
        for batch in range(self.batch_size):
            for head in range(self.num_key_value_heads):
                ranked_mass = block_probability[
                    batch, head, :, :, order[batch, head]
                ].cumsum(-1)
                ranked_output = block_output[
                    batch, head, :, :, order[batch, head], :
                ].cumsum(-2)
                selected_mass = fixed_mass[
                    batch, head, ..., None
                ] + torch.cat((
                    torch.zeros_like(ranked_mass[..., :1]),
                    ranked_mass[..., :maximum],
                ), dim=-1)
                selected_output = fixed_output[
                    batch, head, ..., None, :
                ] + torch.cat((
                    torch.zeros_like(ranked_output[..., :1, :]),
                    ranked_output[..., :maximum, :],
                ), dim=-2)
                sparse = selected_output / selected_mass[
                    ..., None
                ].clamp_min(1e-30)
                delta = (
                    sparse - predicted_full[batch, head, ..., None, :]
                ).permute(2, 0, 1, 3)
                relative = self.head_alloc_mode.endswith("_rel")
                group_loss = delta.norm(dim=-1)
                if relative:
                    reference = predicted_full[
                        batch, head
                    ].norm(dim=-1).clamp_min(1e-6)
                    group_loss = group_loss / reference[None]
                if "_gmax" in self.head_alloc_mode:
                    head_loss = group_loss.flatten(1).amax(-1)
                elif self.head_alloc_mode.endswith("_gmean"):
                    head_loss = group_loss.flatten(1).mean(-1)
                else:
                    flat_delta = delta.flatten(1)
                    if relative:
                        reference = predicted_full[
                            batch, head
                        ].flatten().norm().clamp_min(1e-6)
                        head_loss = flat_delta.norm(dim=-1) / reference
                    else:
                        head_loss = flat_delta.norm(dim=-1)
                if "_wo" in self.head_alloc_mode:
                    wo = self._head_wo_projection[layer_idx]
                    if wo is None:
                        raise RuntimeError(
                            "output-vmean-WO requires the layer output projection"
                        )
                    width = self.num_key_value_groups * self.head_dim
                    wo_head = wo.float()[
                        :, head * width:(head + 1) * width
                    ]
                    head_loss = torch.mm(
                        delta.flatten(1), wo_head.T
                    ).norm(dim=-1)
                if self.head_alloc_mode.endswith("_sq"):
                    head_loss = head_loss.square()
                loss[batch, head] = head_loss
        self._head_allocation_loss = loss

    def _score_blocks_and_candidate_fraction(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        first_block: int,
        last_block: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Also estimate how much mass remains outside exact regions.

        Per-head normalization over candidate blocks destroys an important
        signal: heads already explained by the exact prefix/recent tokens need
        less dynamic budget.  The block summaries provide the candidate LSE;
        only the (small) exact region is dotted with the live query.
        """
        query, block_logits = self._block_logits(
            layer_idx, query_states, first_block, last_block
        )
        self._last_ragged_query = query
        self._last_ragged_block_logits = block_logits
        score = self._reduce_block_logits(block_logits)
        exact_positions = self._exact_position_ids(layer_idx)
        if not exact_positions.numel():
            self._last_ragged_candidate_fraction_groups = torch.ones_like(
                block_logits[..., 0]
            )
            return score, torch.ones_like(score[..., 0])
        source = self.k_cache[layer_idx]
        exact_keys = source.index_select(
            2, exact_positions.to(source.device)
        ).to(self.compute_device, non_blocking=self.offload).float()
        fixed_logits = torch.einsum(
            "bhgqd,bhtd->bhgqt", query.float(), exact_keys
        ) / math.sqrt(self.head_dim)
        fixed_lse = torch.logsumexp(fixed_logits, dim=-1)
        candidate_lse = torch.logsumexp(block_logits.float(), dim=-1)
        fraction_groups = torch.sigmoid(candidate_lse - fixed_lse)
        self._last_ragged_candidate_fraction_groups = fraction_groups
        fraction = fraction_groups.mean(dim=(2, 3))
        if (
            self.head_alloc_mode.startswith("output_vmean")
            or self.head_alloc_mode.startswith("output_sensitivity")
        ):
            value_source = self.v_cache[layer_idx]
            exact_values = value_source.index_select(
                2, exact_positions.to(value_source.device)
            ).to(self.compute_device, non_blocking=self.offload).float()
            joint_lse = torch.logaddexp(fixed_lse, candidate_lse)[..., None]
            fixed_probability = torch.exp(fixed_logits - joint_lse)
            block_probability = torch.exp(block_logits.float() - joint_lse)
            # Actual candidate attention mass per KV head.  The allocator's
            # semantic-preservation term must not reuse ``score`` here:
            # ``score`` may use GQA-max (or an influence interpolation) and
            # therefore is a routing utility rather than a probability
            # measure.  Averaging the normalized probabilities across the
            # query heads/positions gives a tail with a direct interpretation:
            # expected attention mass discarded by this KV head.
            candidate_block_mass = block_probability.mean(dim=(2, 3))
            fixed_mass = fixed_probability.sum(-1)
            fixed_output = torch.einsum(
                "bhgqt,bhtd->bhgqd", fixed_probability, exact_values
            )
            block_value = self.router_value_mean[
                layer_idx, :, :, first_block:last_block
            ].float()
            if self.component_vmean_allocation:
                component_output = self._component_value_block_output(
                    layer_idx,
                    query,
                    joint_lse,
                    first_block,
                    last_block,
                )
                mean_output = block_probability[..., None] * block_value[
                    :, :, None, None
                ]
                block_output = (
                    (1.0 - self.component_vmean_blend) * mean_output
                    + self.component_vmean_blend * component_output
                )
            else:
                block_output = block_probability[..., None] * block_value[
                    :, :, None, None
                ]
            predicted_full = fixed_output + block_output.sum(-2)
            if self.head_alloc_mode.startswith("output_sensitivity"):
                # First-order influence of deleting a block from a normalized
                # weighted mean: p_G (v_G-o)/(1-p_G).  The small-p common
                # denominator is omitted for ranking; GQA query heads are
                # pooled in Euclidean norm.
                influence = block_probability[..., None] * (
                    block_value[:, :, None, None]
                    - predicted_full[..., None, :]
                )
                sensitivity = influence.square().sum(dim=(2, 3, 5)).sqrt()
                beta = {
                    "output_sensitivity": 1.0,
                    "output_sensitivity_b025": 0.25,
                    "output_sensitivity_b05": 0.5,
                    "output_sensitivity_b075": 0.75,
                }[self.head_alloc_mode]
                if beta == 1.0:
                    score = sensitivity
                else:
                    # Geometric interpolation is invariant to arbitrary
                    # positive rescaling of either router score.  beta=0 is
                    # the deployed mass ranking; beta=1 is pure first-order
                    # output influence.
                    score = (
                        score.float().clamp_min(1e-30).pow(1.0 - beta)
                        * sensitivity.clamp_min(1e-30).pow(beta)
                    )
            maximum = min(
                block_logits.shape[-1],
                self.head_alloc_max_budget // self.block_size,
            )
            loss = torch.full(
                (
                    self.batch_size,
                    self.num_key_value_heads,
                    maximum + 1,
                ),
                float("inf"), device=self.compute_device,
            )
            mass_loss = torch.full_like(loss, float("inf"))
            kl_loss = torch.full_like(loss, float("inf"))
            peak_loss = torch.full_like(loss, float("inf"))
            influence_tail_loss = torch.full_like(loss, float("inf"))
            renyi_loss = torch.full_like(loss, float("inf"))
            joint_vectors = (
                [[None for _ in range(self.num_key_value_heads)]
                 for _ in range(self.batch_size)]
                if self.head_alloc_mode == "output_vmean_jointwo"
                else None
            )
            order = score.argsort(-1, descending=True)
            for batch in range(self.batch_size):
                for head in range(self.num_key_value_heads):
                    ranked_mass = block_probability[
                        batch, head, :, :, order[batch, head]
                    ].cumsum(-1)
                    ranked_router = candidate_block_mass[
                        batch, head, order[batch, head]
                    ]
                    router_tail = torch.cat((
                        ranked_router.sum()[None],
                        (
                            ranked_router.sum()
                            - ranked_router.cumsum(-1)[..., :maximum]
                        ).clamp_min(0),
                    ))
                    router_tail_l2 = torch.cat((
                        torch.flip(
                            torch.cumsum(
                                torch.flip(ranked_router.square(), dims=(-1,)),
                                dim=-1,
                            ),
                            dims=(-1,),
                        ).sqrt(),
                        torch.zeros_like(ranked_router[:1]),
                    ))[:maximum + 1]
                    ranked_probability = block_probability[
                        batch, head, :, :, order[batch, head]
                    ]
                    # Minkowski upper proxy for the omitted-output numerator.
                    # For approximate full output o, omitting block i adds
                    # p_i (o-v_i); summing the norms prevents cancellation
                    # from hiding a rare but semantically distinct value.
                    influence = block_probability[
                        batch, head, ..., None
                    ] * (
                        predicted_full[batch, head, ..., None, :]
                        - block_value[batch, head, None, None]
                    )
                    block_influence = influence.square().sum(
                        dim=(0, 1, 3)
                    ).sqrt()
                    ranked_influence = block_influence[order[batch, head]]
                    influence_tail_curve = torch.cat((
                        torch.flip(
                            torch.cumsum(
                                torch.flip(ranked_influence, dims=(-1,)),
                                dim=-1,
                            ),
                            dims=(-1,),
                        ),
                        torch.zeros_like(ranked_influence[:1]),
                    ))[:maximum + 1]
                    suffix_peak = torch.flip(
                        torch.cummax(
                            torch.flip(ranked_probability, dims=(-1,)), dim=-1
                        ).values,
                        dims=(-1,),
                    )
                    peak_curve = torch.cat((
                        suffix_peak,
                        torch.zeros_like(suffix_peak[..., :1]),
                    ), dim=-1)[..., :maximum + 1]
                    ranked_output = block_output[
                        batch, head, :, :, order[batch, head], :
                    ].cumsum(-2)
                    zero_mass = torch.zeros_like(ranked_mass[..., :1])
                    zero_output = torch.zeros_like(ranked_output[..., :1, :])
                    selected_mass = fixed_mass[
                        batch, head, ..., None
                    ] + torch.cat(
                        (zero_mass, ranked_mass[..., :maximum]), dim=-1
                    )
                    selected_output = fixed_output[
                        batch, head, ..., None, :
                    ] + torch.cat(
                        (zero_output, ranked_output[..., :maximum, :]), dim=-2
                    )
                    sparse = selected_output / selected_mass[
                        ..., None
                    ].clamp_min(1e-30)
                    # Conditioning dense attention on a retained set S has
                    # exact reverse-KL regret -log p(S).  This supplies a
                    # semantic coverage objective that cannot be cancelled by
                    # opposing value vectors.  The minimax variant protects a
                    # single GQA query head instead of averaging it away.
                    group_kl = -selected_mass.clamp_min(1e-30).log()
                    if self.head_alloc_mode == "output_vmean_klmax01":
                        kl_loss[batch, head] = group_kl.flatten(0, 1).amax(0)
                    else:
                        kl_loss[batch, head] = group_kl.mean(dim=(0, 1))
                    delta = sparse - predicted_full[
                        batch, head, ..., None, :
                    ]
                    # [count, GQA-group, decode-query, value-dim].  The
                    # default Euclidean objective pools all query heads.  The
                    # gmax variant is a minimax allocation: one semantic query
                    # head cannot be hidden by three easy siblings sharing the
                    # same KV head.  gmean provides the corresponding L1
                    # control so this is an interpretable p-norm comparison.
                    delta = delta.permute(2, 0, 1, 3)
                    # Keep the structured [count, GQA, query, value] view for
                    # the joint post-W_O objective below.  Some separable
                    # objectives flatten it, but W_O must concatenate only
                    # the GQA value channels belonging to the same query.
                    delta_grouped = delta
                    relative = self.head_alloc_mode.endswith("_rel")
                    group_loss = delta.norm(dim=-1)
                    if relative:
                        reference = predicted_full[
                            batch, head
                        ].norm(dim=-1).clamp_min(1e-6)
                        group_loss = group_loss / reference[None]
                    if "_gmax" in self.head_alloc_mode:
                        head_loss = group_loss.flatten(1).amax(-1)
                    elif self.head_alloc_mode == "output_vmean_gp4":
                        # Smooth distributionally-robust objective between
                        # the default pooled L2 error and the brittle GQA
                        # maximum.  A large error in one semantic query head
                        # cannot disappear inside several easy siblings.
                        head_loss = (
                            group_loss.flatten(1).pow(4).mean(-1)
                        ).pow(0.25)
                    elif self.head_alloc_mode == "output_vmean_cvar50":
                        # CVaR over the worst half of the GQA query heads.
                        # Unlike max-risk, this is stable to one anomalous
                        # head while still controlling the semantic tail.
                        flat_group_loss = group_loss.flatten(1)
                        tail = max(1, (flat_group_loss.shape[-1] + 1) // 2)
                        head_loss = flat_group_loss.topk(tail, dim=-1).values.mean(-1)
                    elif self.head_alloc_mode.endswith("_gmean"):
                        head_loss = group_loss.flatten(1).mean(-1)
                    else:
                        delta = delta.flatten(1)
                        if relative:
                            reference = predicted_full[
                                batch, head
                            ].flatten().norm().clamp_min(1e-6)
                            head_loss = delta.norm(dim=-1) / reference
                        else:
                            head_loss = None
                    if self.head_alloc_mode == "output_vmean_jointwo":
                        wo = self._head_wo_projection[layer_idx]
                        if wo is None:
                            raise RuntimeError(
                                "joint post-W_O allocation requires the "
                                "layer output projection"
                            )
                        width = self.num_key_value_groups * self.head_dim
                        wo_head = wo.float()[
                            :, head * width:(head + 1) * width,
                        ]
                        # Preserve query positions while joining the GQA
                        # query heads represented by this KV head.
                        projected = torch.matmul(
                            delta_grouped.permute(0, 2, 1, 3).flatten(2),
                            wo_head.T,
                        ).flatten(1)
                        joint_vectors[batch][head] = projected
                        head_loss = projected.square().sum(-1)
                    elif "_wo" in self.head_alloc_mode:
                        wo = self._head_wo_projection[layer_idx]
                        if wo is None:
                            raise RuntimeError(
                                "output-vmean-WO requires the layer output projection"
                            )
                        width = self.num_key_value_groups * self.head_dim
                        wo_head = wo.float()[
                            :,
                            head * width:(head + 1) * width,
                        ]
                        delta = torch.mm(delta, wo_head.T)
                        head_loss = delta.norm(dim=-1)
                    elif head_loss is None:
                        head_loss = delta.norm(dim=-1)
                    if self.head_alloc_mode.endswith("_sq"):
                        head_loss = head_loss.square()
                    loss[batch, head] = head_loss
                    mass_loss[batch, head] = router_tail
                    peak_loss[batch, head] = peak_curve.flatten(0, 1).amax(0)
                    influence_tail_loss[batch, head] = influence_tail_curve
                    renyi_loss[batch, head] = router_tail_l2
            if joint_vectors is not None:
                self._head_allocation_joint_vectors = torch.stack([
                    torch.stack(per_head) for per_head in joint_vectors
                ])
            else:
                self._head_allocation_joint_vectors = None
            if self.head_alloc_mode in {
                "output_vmean_bal2", "output_vmean_bal4"
            }:
                # Distributionally balanced allocation.  Raw sum-error may
                # sacrifice one KV head to improve several easy heads.  Scale
                # every curve by its own uniform-budget loss, then choose an
                # Lp point between total-error minimization and minimax.
                uniform = min(self.select_blocks, maximum)
                relative_curve = loss / loss[
                    ..., uniform, None
                ].clamp_min(1e-8)
                power = 2 if self.head_alloc_mode.endswith("bal2") else 4
                loss = relative_curve.pow(power)
            if self.head_alloc_mode in {
                "output_vmean_mass01", "output_vmean_mass025"
            }:
                weight = (
                    0.1 if self.head_alloc_mode.endswith("mass01") else 0.25
                )
                uniform = min(self.select_blocks, maximum)
                output_scale = loss[..., uniform].mean(
                    -1, keepdim=True
                )
                mass_scale = mass_loss[..., uniform].mean(
                    -1, keepdim=True
                ).clamp_min(1e-8)
                loss = loss + (
                    weight
                    * (output_scale / mass_scale)[..., None]
                    * mass_loss
                )
            elif self.head_alloc_mode in {
                "output_vmean_kl01", "output_vmean_kl05",
                "output_vmean_kl1", "output_vmean_klmax01",
            }:
                kl_weight = {
                    "output_vmean_kl01": 0.1,
                    "output_vmean_kl05": 0.5,
                    "output_vmean_kl1": 1.0,
                    "output_vmean_klmax01": 0.1,
                }[self.head_alloc_mode]
                uniform = min(self.select_blocks, maximum)
                output_scale = loss[..., uniform].mean(-1, keepdim=True)
                kl_scale = kl_loss[..., uniform].mean(
                    -1, keepdim=True
                ).clamp_min(1e-8)
                loss = loss + (
                    kl_weight
                    * (output_scale / kl_scale)[..., None]
                    * kl_loss
                )
            elif self.head_alloc_mode in {
                "output_vmean_peak005",
                "output_vmean_peak005_powerprior",
            }:
                uniform = min(self.select_blocks, maximum)
                output_scale = loss[..., uniform].mean(-1, keepdim=True)
                peak_scale = peak_loss[..., uniform].mean(
                    -1, keepdim=True
                ).clamp_min(1e-8)
                loss = loss + (
                    self.head_alloc_peak_weight
                    * (output_scale / peak_scale)[..., None]
                    * peak_loss
                )
            elif self.head_alloc_mode == "output_vmean_peakconc":
                uniform = min(self.select_blocks, maximum)
                output_scale = loss[..., uniform].mean(-1, keepdim=True)
                peak_scale = peak_loss[..., uniform].mean(
                    -1, keepdim=True
                ).clamp_min(1e-8)
                # Activate singleton-tail protection only where it is needed.
                # P/R is the concentration of the omitted mass at the
                # uniform operating point: it is near one for a one-event
                # semantic tail and small for a diffuse tail.  Mean
                # normalization preserves the interpretation of the global
                # peak weight while adapting it across KV heads.
                concentration = (
                    peak_loss[..., uniform]
                    / mass_loss[..., uniform].clamp_min(1e-8)
                )
                concentration = concentration / concentration.mean(
                    -1, keepdim=True
                ).clamp_min(1e-8)
                loss = loss + (
                    self.head_alloc_peak_weight
                    * (output_scale / peak_scale)[..., None]
                    * concentration[..., None]
                    * peak_loss
                )
            elif self.head_alloc_mode == "output_vmean_peaktail":
                uniform = min(self.select_blocks, maximum)
                output_scale = loss[..., uniform].mean(-1, keepdim=True)
                peak_scale = peak_loss[..., uniform].mean(
                    -1, keepdim=True
                ).clamp_min(1e-8)
                tail_scale = influence_tail_loss[..., uniform].mean(
                    -1, keepdim=True
                ).clamp_min(1e-8)
                loss = loss + (
                    self.head_alloc_peak_weight
                    * (output_scale / peak_scale)[..., None]
                    * peak_loss
                    + self.head_alloc_tail_weight
                    * (output_scale / tail_scale)[..., None]
                    * influence_tail_loss
                )
            elif self.head_alloc_mode == "output_vmean_masspeak":
                uniform = min(self.select_blocks, maximum)
                output_scale = loss[..., uniform].mean(-1, keepdim=True)
                mass_scale = mass_loss[..., uniform].mean(
                    -1, keepdim=True
                ).clamp_min(1e-8)
                peak_scale = peak_loss[..., uniform].mean(
                    -1, keepdim=True
                ).clamp_min(1e-8)
                loss = loss + (
                    0.1
                    * (output_scale / mass_scale)[..., None]
                    * mass_loss
                    + 0.05
                    * (output_scale / peak_scale)[..., None]
                    * peak_loss
                )
            elif self.head_alloc_mode == "output_vmean_renyi025":
                uniform = min(self.select_blocks, maximum)
                output_scale = loss[..., uniform].mean(-1, keepdim=True)
                renyi_scale = renyi_loss[..., uniform].mean(
                    -1, keepdim=True
                ).clamp_min(1e-8)
                loss = loss + (
                    0.25
                    * (output_scale / renyi_scale)[..., None]
                    * renyi_loss
                )
            self._head_allocation_loss = loss
        return score, fraction

    def _ragged_position_ids(
        self,
        layer_idx: int,
        scores: torch.Tensor,
        first_block: int,
        candidate_fraction: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.refine_tokens:
            if self.refine_factor == 1.0 and self.refine_candidate_ratio is None:
                allocation_scores = None
                coarse_logits = self._last_ragged_block_logits
                if self.head_alloc_mode in {
                    "entropy_mixture", "entropy_mixture_residual",
                    "coverage_mixture",
                }:
                    if coarse_logits is None:
                        raise RuntimeError(
                            "GQA-mixture allocation lost its router logits"
                        )
                    # Average normalized sibling distributions, rather than
                    # taking entropy after a pointwise max.  Block ranking
                    # remains GQA-max; this changes only the per-KV-head B_h.
                    allocation_scores = torch.softmax(
                        coarse_logits.float(), dim=-1
                    ).mean(dim=(2, 3))
                if self.head_alloc_mode == "coverage_worst":
                    if coarse_logits is None:
                        raise RuntimeError(
                            "worst-GQA coverage lost its router logits"
                        )
                    if self.query_group_mean:
                        raise RuntimeError(
                            "worst-GQA coverage requires individual GQA queries"
                        )
                    fraction_groups = (
                        self._last_ragged_candidate_fraction_groups
                    )
                    if fraction_groups is None:
                        raise RuntimeError(
                            "worst-GQA coverage lost exact-region mass"
                        )
                    probability = torch.softmax(
                        coarse_logits.float(), dim=-1
                    )
                    order = scores.argsort(dim=-1, descending=True)
                    gather_order = order[:, :, None, None].expand_as(
                        probability
                    )
                    cumulative = probability.gather(
                        -1, gather_order
                    ).cumsum(-1)
                    required = (
                        1.0
                        - (1.0 - self.head_alloc_coverage)
                        / fraction_groups.float().clamp_min(1e-30)
                    ).clamp(0, 1)
                    per_query = (
                        cumulative < required[..., None]
                    ).sum(-1)
                    per_query = per_query + (required > 0).to(
                        per_query.dtype
                    )
                    counts = per_query.amax(dim=(2, 3))
                    available = scores.shape[-1]
                    minimum = min(
                        self.head_alloc_min_budget // self.block_size,
                        available,
                    )
                    maximum = min(
                        self.head_alloc_max_budget // self.block_size,
                        available,
                    )
                    counts = counts.clamp(min=minimum, max=maximum)
                    achieved = cumulative.gather(
                        -1,
                        (counts[:, :, None, None] - 1).clamp_min(0)[
                            ..., None
                        ].expand(*cumulative.shape[:-1], 1),
                    ).squeeze(-1)
                    achieved = torch.where(
                        counts[:, :, None, None] > 0,
                        achieved,
                        torch.zeros_like(achieved),
                    )
                    predicted = (
                        1.0 - fraction_groups
                        + fraction_groups * achieved
                    ).amin(dim=(2, 3))
                    self._head_allocation_history.append(
                        (counts * self.block_size).detach().cpu()
                    )
                    self._head_predicted_coverage_history.append(
                        predicted.detach().cpu()
                    )
                    return self._ragged_positions_from_counts(
                        layer_idx, scores, counts, first_block
                    )
                return super()._ragged_position_ids(
                    layer_idx,
                    scores,
                    first_block,
                    candidate_fraction,
                    allocation_scores=allocation_scores,
                )
            query = self._last_ragged_query
            coarse_logits = self._last_ragged_block_logits
            if query is None or coarse_logits is None:
                raise RuntimeError("block refinement lost its ragged router state")
            available = scores.shape[-1]
            if self.refine_candidate_ratio is None:
                requested = int(math.ceil(
                    self.refine_factor * self.select_blocks
                ))
            else:
                requested = int(math.ceil(
                    self.refine_candidate_ratio * available
                ))
            # The candidate pool must accommodate a head that receives the
            # active cap, otherwise the allocator can request a block whose
            # exact score was never computed.
            cap = min(
                available, self.head_alloc_max_budget // self.block_size
            )
            candidate_count = min(available, max(requested, cap))
            candidate_relative = scores.topk(
                candidate_count, dim=-1
            ).indices
            candidate_blocks = candidate_relative + first_block
            keys = self._gather_candidate_keys(layer_idx, candidate_blocks)
            exact_token_logits = torch.einsum(
                "bhgqd,bhcsd->bhgqcs", query.float(), keys.float()
            ) / math.sqrt(self.head_dim)
            exact_block_logits = torch.logsumexp(exact_token_logits, dim=-1)
            hybrid_logits = coarse_logits.clone()
            scatter_index = candidate_relative[:, :, None, None].expand(
                self.batch_size,
                self.num_key_value_heads,
                hybrid_logits.shape[2],
                hybrid_logits.shape[3],
                candidate_count,
            )
            hybrid_logits.scatter_(-1, scatter_index, exact_block_logits)
            refined_scores = self._reduce_block_logits(hybrid_logits)
            self._set_output_vmean_loss_for_ranking(
                layer_idx,
                query,
                hybrid_logits,
                refined_scores,
                first_block,
                first_block + available,
            )
            return super()._ragged_position_ids(
                layer_idx, refined_scores, first_block, candidate_fraction
            )
        query = self._last_ragged_query
        coarse_logits = self._last_ragged_block_logits
        if query is None or coarse_logits is None:
            raise RuntimeError("token refinement lost its ragged router state")
        allocation_scores = scores
        if self.head_alloc_mode in {
            "entropy_mixture", "entropy_mixture_residual"
        }:
            # Each GQA sibling induces a probability distribution over
            # blocks.  Their arithmetic mixture has entropy
            # H(mean_g p_g) = mean_g H(p_g) + JS({p_g}), so it charges both
            # within-query diffuseness and disagreement among the query heads
            # sharing this KV head.  In contrast, entropy after a pointwise
            # max is not the entropy of any mixture distribution.
            allocation_scores = torch.softmax(
                coarse_logits.float(), dim=-1
            ).mean(dim=(2, 3))
        counts = self._allocate_head_blocks(
            layer_idx, allocation_scores, candidate_fraction
        )
        if self.component_candidate_refinement:
            return self._component_candidate_position_ids(
                layer_idx, query, scores, counts, first_block
            )
        available = scores.shape[-1]
        if self.refine_candidate_ratio is None:
            requested = int(math.ceil(self.refine_factor * counts.max().item()))
        else:
            requested = int(math.ceil(
                self.refine_candidate_ratio * available
            ))
        allocation_cap = min(
            available, self.head_alloc_max_budget // self.block_size
        )
        candidate_count = min(
            available,
            max(int(counts.max().item()), requested, allocation_cap),
        )
        candidate_relative = scores.topk(candidate_count, dim=-1).indices
        candidate_blocks = candidate_relative + first_block
        if self.refine_sketch_rank:
            exact_token_logits = self._refine_sketch_logits(
                layer_idx, query, candidate_blocks
            )
        else:
            keys = self._gather_candidate_keys(layer_idx, candidate_blocks)
            exact_token_logits = torch.einsum(
                "bhgqd,bhcsd->bhgqcs", query.float(), keys.float()
            ) / math.sqrt(self.head_dim)
        exact_block_logits = torch.logsumexp(exact_token_logits, dim=-1)
        hybrid_logits = coarse_logits.clone()
        scatter_index = candidate_relative[:, :, None, None].expand(
            self.batch_size,
            self.num_key_value_heads,
            hybrid_logits.shape[2],
            hybrid_logits.shape[3],
            candidate_count,
        )
        hybrid_logits.scatter_(-1, scatter_index, exact_block_logits)
        if self.head_alloc_mode == "entropy_refined":
            # The first allocation formed a sufficiently large candidate
            # pool.  Re-estimate effective support from the hybrid router,
            # which is exact on that pool, before committing the per-head
            # budgets.  This reuses logits already required by token reranking
            # and therefore adds no KV traffic.
            refined_scores = self._reduce_block_logits(hybrid_logits)
            # Replace, rather than double-count, the provisional allocation
            # in the diagnostic stream.
            if self._head_allocation_history:
                self._head_allocation_history.pop()
            counts = self._allocate_head_blocks(
                layer_idx, refined_scores, candidate_fraction
            )
        log_normalizer = torch.logsumexp(
            hybrid_logits.float(), dim=-1, keepdim=True
        )
        token_probability_for_ranking = (
            exact_token_logits - log_normalizer[..., None]
        ).exp()
        if self.refine_token_gqa_reduce == "max":
            token_score = token_probability_for_ranking.amax(
                dim=(2, 3)
            ).flatten(-2)
        elif self.refine_token_gqa_reduce in {"p2", "p4"}:
            power = 2 if self.refine_token_gqa_reduce == "p2" else 4
            token_score = token_probability_for_ranking.pow(power).mean(
                dim=(2, 3)
            ).pow(1.0 / power).flatten(-2)
        elif self.refine_token_gqa_reduce == "ucb":
            # Distributionally robust aggregation over the query heads that
            # share one KV head.  Mean is optimal for the nominal uniform
            # mixture; the standard-deviation bonus protects evidence used by
            # only one sibling without applying a hard max everywhere.
            mean_probability = token_probability_for_ranking.mean(dim=(2, 3))
            variance = (
                token_probability_for_ranking
                - mean_probability[:, :, None, None]
            ).square().mean(dim=(2, 3))
            token_score = (
                mean_probability
                + self.refine_token_gqa_ucb * variance.sqrt()
            ).flatten(-2)
        else:
            token_score = token_probability_for_ranking.mean(
                dim=(2, 3)
            ).flatten(-2)
        if self.match_token_allocation:
            self._set_output_vmean_loss_for_token_ranking(
                layer_idx,
                query,
                hybrid_logits,
                candidate_blocks,
                exact_token_logits,
                token_score,
            )
            counts = self._allocate_head_blocks(
                layer_idx, scores, candidate_fraction
            )
        maximum_tokens = int(counts.max().item()) * self.block_size
        winner = token_score.topk(maximum_tokens, dim=-1).indices
        offsets = torch.arange(
            self.block_size,
            device=candidate_blocks.device,
            dtype=candidate_blocks.dtype,
        )
        candidate_positions = (
            candidate_blocks[..., None] * self.block_size + offsets
        ).flatten(-2)
        ranked_positions = candidate_positions.gather(-1, winner)
        exact = self._exact_position_ids(layer_idx)
        exact_count = int(exact.numel())
        lengths = counts * self.block_size + exact_count
        width = int(lengths.max().item())
        positions = torch.zeros(
            self.batch_size, self.num_key_value_heads, width,
            device=scores.device, dtype=torch.long,
        )
        for batch in range(self.batch_size):
            for head in range(self.num_key_value_heads):
                dynamic = int(counts[batch, head].item()) * self.block_size
                positions[batch, head, :dynamic] = ranked_positions[
                    batch, head, :dynamic
                ]
                if exact_count:
                    positions[
                        batch, head, dynamic:dynamic + exact_count
                    ] = exact
        return positions.contiguous(), lengths.to(torch.int32).contiguous()

    def _component_candidate_position_ids(
        self,
        layer_idx: int,
        query: torch.Tensor,
        scores: torch.Tensor,
        counts: torch.Tensor,
        first_block: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shortlist singleton components, then rerank exact candidate tokens.

        This diagnostic deliberately requires full order (r=S). It isolates
        the loss caused by collapsing exact token components into one score per
        temporal block before shortlist construction.
        """
        component_logits = self._last_ragged_component_logits
        if component_logits is None:
            raise RuntimeError("component candidate logits are unavailable")
        available_blocks = scores.shape[-1]
        block_counts = self.component_count[
            layer_idx, :, :, first_block:first_block + available_blocks
        ]
        if torch.any(block_counts != self.block_size):
            raise RuntimeError(
                "component candidates require S components in every candidate block"
            )
        if self.router_slots != self.block_size:
            raise RuntimeError("component candidates require max_components=S")

        maximum_tokens = int(counts.max().item()) * self.block_size
        available_tokens = available_blocks * self.block_size
        if self.refine_candidate_ratio is None:
            requested = int(math.ceil(self.refine_factor * maximum_tokens))
        else:
            requested = int(math.ceil(
                self.refine_candidate_ratio * available_tokens
            ))
        candidate_count = min(
            available_tokens, max(maximum_tokens, requested)
        )
        component_score = self._reduce_component_logits(component_logits)
        flat_candidate = component_score.flatten(-2).topk(
            candidate_count, dim=-1
        ).indices
        candidate_positions = first_block * self.block_size + flat_candidate

        source = self.k_cache[layer_idx]
        source_positions = candidate_positions.to(source.device)
        index = source_positions[..., None].expand(
            *source_positions.shape, self.head_dim
        )
        keys = source.gather(2, index).to(
            self.compute_device, non_blocking=self.offload
        )
        exact_token_logits = torch.einsum(
            "bhgqd,bhcd->bhgqc", query.float(), keys.float()
        ) / math.sqrt(self.head_dim)

        candidate_lse = torch.logsumexp(
            component_logits.float().flatten(-2), dim=-1
        )
        exact = self._exact_position_ids(layer_idx)
        if exact.numel():
            exact_keys = source.index_select(
                2, exact.to(source.device)
            ).to(self.compute_device, non_blocking=self.offload).float()
            fixed_logits = torch.einsum(
                "bhgqd,bhtd->bhgqt", query.float(), exact_keys
            ) / math.sqrt(self.head_dim)
            joint_lse = torch.logaddexp(
                candidate_lse, torch.logsumexp(fixed_logits, dim=-1)
            )
        else:
            joint_lse = candidate_lse
        probability = torch.exp(
            exact_token_logits.float() - joint_lse[..., None]
        )
        if self.refine_token_gqa_reduce == "max":
            token_score = probability.amax(dim=(2, 3))
        elif self.refine_token_gqa_reduce in {"p2", "p4"}:
            power = 2 if self.refine_token_gqa_reduce == "p2" else 4
            token_score = probability.pow(power).mean(
                dim=(2, 3)
            ).pow(1.0 / power)
        elif self.refine_token_gqa_reduce == "ucb":
            mean = probability.mean(dim=(2, 3))
            variance = (
                probability - mean[:, :, None, None]
            ).square().mean(dim=(2, 3))
            token_score = mean + self.refine_token_gqa_ucb * variance.sqrt()
        else:
            token_score = probability.mean(dim=(2, 3))

        winner = token_score.topk(maximum_tokens, dim=-1).indices
        ranked_positions = candidate_positions.gather(-1, winner)
        exact_count = int(exact.numel())
        lengths = counts * self.block_size + exact_count
        width = int(lengths.max().item())
        positions = torch.zeros(
            self.batch_size,
            self.num_key_value_heads,
            width,
            device=scores.device,
            dtype=torch.long,
        )
        for batch in range(self.batch_size):
            for head in range(self.num_key_value_heads):
                dynamic = int(counts[batch, head].item()) * self.block_size
                positions[batch, head, :dynamic] = ranked_positions[
                    batch, head, :dynamic
                ]
                if exact_count:
                    positions[
                        batch, head, dynamic:dynamic + exact_count
                    ] = exact
        return positions.contiguous(), lengths.to(torch.int32).contiguous()

    def _set_output_vmean_loss_for_token_ranking(
        self,
        layer_idx: int,
        query: torch.Tensor,
        hybrid_block_logits: torch.Tensor,
        candidate_blocks: torch.Tensor,
        exact_token_logits: torch.Tensor,
        token_score: torch.Tensor,
    ) -> None:
        """Build DP curves for the exact token order actually gathered.

        Coarse allocation is used only to form a candidate pool.  This second
        curve makes the final resource allocation consistent with the
        top-token refinement order while retaining coarse mean-V summaries for
        non-candidate blocks.
        """
        if not self.head_alloc_mode.startswith("output_vmean"):
            raise RuntimeError(
                "matched token allocation currently requires output-vmean"
            )
        exact_positions = self._exact_position_ids(layer_idx)
        key_source = self.k_cache[layer_idx]
        value_source = self.v_cache[layer_idx]
        exact_keys = key_source.index_select(
            2, exact_positions.to(key_source.device)
        ).to(self.compute_device, non_blocking=self.offload).float()
        exact_values = value_source.index_select(
            2, exact_positions.to(value_source.device)
        ).to(self.compute_device, non_blocking=self.offload).float()
        fixed_logits = torch.einsum(
            "bhgqd,bhtd->bhgqt", query.float(), exact_keys
        ) / math.sqrt(self.head_dim)
        fixed_lse = torch.logsumexp(fixed_logits, dim=-1)
        candidate_lse = torch.logsumexp(
            hybrid_block_logits.float(), dim=-1
        )
        joint_lse = torch.logaddexp(fixed_lse, candidate_lse)[..., None]
        fixed_probability = torch.exp(fixed_logits - joint_lse)
        block_probability = torch.exp(
            hybrid_block_logits.float() - joint_lse
        )
        token_probability = torch.exp(
            exact_token_logits.float() - joint_lse[..., None]
        )
        fixed_mass = fixed_probability.sum(-1)
        fixed_output = torch.einsum(
            "bhgqt,bhtd->bhgqd", fixed_probability, exact_values
        )

        first_block, last_block = self.block_state[
            layer_idx
        ].candidate_block_range
        block_value = self.router_value_mean[
            layer_idx, :, :, first_block:last_block
        ].float()
        approximate_block_output = block_probability[..., None] * block_value[
            :, :, None, None
        ]
        candidate_relative = candidate_blocks - first_block
        gather_logits = candidate_relative[:, :, None, None, :, None].expand(
            self.batch_size,
            self.num_key_value_heads,
            block_probability.shape[2],
            block_probability.shape[3],
            candidate_blocks.shape[-1],
            self.head_dim,
        )
        candidate_approximate_output = approximate_block_output.gather(
            -2, gather_logits
        )
        candidate_values = self._gather_candidate_values(
            layer_idx, candidate_blocks
        ).float()
        candidate_exact_output = torch.einsum(
            "bhgqcs,bhcsd->bhgqcd",
            token_probability,
            candidate_values,
        )
        predicted_full = (
            fixed_output
            + approximate_block_output.sum(-2)
            - candidate_approximate_output.sum(-2)
            + candidate_exact_output.sum(-2)
        )

        flat_probability = token_probability.flatten(-2)
        flat_values = candidate_values.flatten(-3, -2)
        maximum = min(
            self.head_alloc_max_budget // self.block_size,
            flat_probability.shape[-1] // self.block_size,
        )
        loss = torch.full(
            (self.batch_size, self.num_key_value_heads, maximum + 1),
            float("inf"), device=self.compute_device,
        )
        order = token_score.argsort(-1, descending=True)
        for batch in range(self.batch_size):
            for head in range(self.num_key_value_heads):
                ranked_probability = flat_probability[
                    batch, head, :, :, order[batch, head]
                ]
                ranked_values = flat_values[
                    batch, head, order[batch, head]
                ]
                cumulative_mass = ranked_probability.cumsum(-1)
                cumulative_output = torch.einsum(
                    "gqt,td->gqtd",
                    ranked_probability,
                    ranked_values,
                ).cumsum(-2)
                step_mass = cumulative_mass[
                    ..., self.block_size - 1::self.block_size
                ][..., :maximum]
                step_output = cumulative_output[
                    ..., self.block_size - 1::self.block_size, :
                ][..., :maximum, :]
                selected_mass = fixed_mass[
                    batch, head, ..., None
                ] + torch.cat((
                    torch.zeros_like(step_mass[..., :1]), step_mass
                ), dim=-1)
                selected_output = fixed_output[
                    batch, head, ..., None, :
                ] + torch.cat((
                    torch.zeros_like(step_output[..., :1, :]), step_output
                ), dim=-2)
                sparse = selected_output / selected_mass[
                    ..., None
                ].clamp_min(1e-30)
                delta = (
                    sparse - predicted_full[batch, head, ..., None, :]
                ).permute(2, 0, 1, 3).flatten(1)
                loss[batch, head] = delta.norm(dim=-1)
        self._head_allocation_loss = loss

    def _gather_candidate_values(
        self, layer_idx: int, block_ids: torch.Tensor
    ) -> torch.Tensor:
        """Load exact V for head-specific candidate blocks only."""
        offsets = torch.arange(
            self.block_size, device=block_ids.device, dtype=block_ids.dtype
        )
        positions = (block_ids[..., None] * self.block_size + offsets).flatten(-2)
        source = self.v_cache[layer_idx]
        source_positions = positions.to(source.device)
        index = source_positions[..., None].expand(
            *source_positions.shape, self.head_dim
        )
        values = source.gather(2, index)
        values = values.to(self.compute_device, non_blocking=self.offload)
        return values.view(
            self.batch_size,
            self.num_key_value_heads,
            block_ids.shape[-1],
            self.block_size,
            self.head_dim,
        )

    def _gather_candidate_keys(
        self, layer_idx: int, block_ids: torch.Tensor
    ) -> torch.Tensor:
        """Load exact K for head-specific candidate blocks only."""
        offsets = torch.arange(
            self.block_size, device=block_ids.device, dtype=block_ids.dtype
        )
        positions = (
            block_ids[..., None] * self.block_size + offsets
        ).flatten(-2)
        source = self.k_cache[layer_idx]
        source_positions = positions.to(source.device)
        index = source_positions[..., None].expand(
            *source_positions.shape, self.head_dim
        )
        keys = source.gather(2, index)
        keys = keys.to(self.compute_device, non_blocking=self.offload)
        return keys.view(
            self.batch_size,
            self.num_key_value_heads,
            block_ids.shape[-1],
            self.block_size,
            self.head_dim,
        )

    def _select_block_ids(
        self, layer_idx: int, query_states: torch.Tensor
    ) -> torch.Tensor:
        """Optionally rerank a coarse candidate set by live-query exact LSE."""
        first, last = self.block_state[layer_idx].candidate_block_range
        if last - first <= self.select_blocks:
            # Warm-up is exact: before the candidate pool exceeds the target
            # budget, there is nothing to rank or refine.
            return super()._select_block_ids(layer_idx, query_states)
        if self.refine_factor == 1.0 and self.refine_candidate_ratio is None:
            # Query-mean decode has exactly one routing logit per KV head and
            # block.  The base selector used to materialize a full softmax
            # before top-k, although softmax is strictly monotone and cannot
            # change this ranking.  Skip that normalization on the canonical
            # one-shot path; probability-valued consumers (refinement and
            # allocation diagnostics) continue through their existing code.
            if self.query_group_mean and query_states.shape[-2] == 1:
                self.incoming_q_len = 1
                first, last = self.block_state[layer_idx].candidate_block_range
                available = last - first
                if available < self.select_blocks:
                    raise RuntimeError(
                        f"only {available} active blocks for budget "
                        f"{self.select_blocks}"
                    )
                _, logits = self._block_logits(
                    layer_idx, query_states, first, last
                )
                score = logits.squeeze(2).squeeze(2)
                relative = torch.topk(
                    score, k=self.select_blocks, dim=-1
                ).indices
                return relative + first
            return super()._select_block_ids(layer_idx, query_states)

        self.incoming_q_len = query_states.shape[-2]
        first, last = self.block_state[layer_idx].candidate_block_range
        available = last - first
        if available < self.select_blocks:
            raise RuntimeError(
                f"only {available} active blocks for budget {self.select_blocks}"
            )
        query, coarse_logits = self._block_logits(
            layer_idx, query_states, first, last
        )
        coarse_score = self._reduce_block_logits(coarse_logits)
        if self.refine_candidate_ratio is None:
            requested_candidates = int(math.ceil(
                self.refine_factor * self.select_blocks
            ))
        else:
            # ParisKV reranks a fixed fraction of the currently retrievable
            # tokens.  Because this router preserves complete block-8 units,
            # the same traffic is matched up to at most block_size-1 tokens.
            requested_candidates = int(math.ceil(
                self.refine_candidate_ratio * available
            ))
        candidate_count = min(
            available, max(self.select_blocks, requested_candidates)
        )
        candidate_relative = coarse_score.topk(
            candidate_count, dim=-1
        ).indices
        candidate_blocks = candidate_relative + first
        keys = self._gather_candidate_keys(layer_idx, candidate_blocks)
        exact_logits = torch.einsum(
            "bhgqd,bhcsd->bhgqcs", query.float(), keys.float()
        ) / math.sqrt(self.head_dim)
        exact_block_logits = torch.logsumexp(exact_logits, dim=-1)

        # Correct the coarse per-query-head normalizer with exact candidate
        # logits, while leaving non-candidates represented by their summaries.
        hybrid_logits = coarse_logits.clone()
        scatter_index = candidate_relative[:, :, None, None].expand(
            self.batch_size,
            self.num_key_value_heads,
            hybrid_logits.shape[2],
            hybrid_logits.shape[3],
            candidate_count,
        )
        hybrid_logits.scatter_(-1, scatter_index, exact_block_logits)
        hybrid_probability = torch.softmax(hybrid_logits.float(), dim=-1)
        # Total block mass is the mean probability across all query heads and
        # query positions sharing a KV head.
        hybrid_score = hybrid_probability.mean(dim=(2, 3))
        candidate_score = hybrid_score.gather(-1, candidate_relative)
        winner_in_candidate = candidate_score.topk(
            self.select_blocks, dim=-1
        ).indices
        return candidate_blocks.gather(-1, winner_in_candidate)

    def get_retrieval_position_ids(
        self, layer_idx: int, query_states: torch.Tensor
    ) -> torch.Tensor:
        if not self.refine_tokens:
            return super().get_retrieval_position_ids(layer_idx, query_states)

        self.incoming_q_len = query_states.shape[-2]
        first, last = self.block_state[layer_idx].candidate_block_range
        available = last - first
        if available < self.select_blocks:
            raise RuntimeError(
                f"only {available} active blocks for budget {self.select_blocks}"
            )
        query, coarse_logits = self._block_logits(
            layer_idx, query_states, first, last
        )
        coarse_score = self._reduce_block_logits(coarse_logits)
        if self.component_candidate_refinement:
            counts = torch.full(
                (self.batch_size, self.num_key_value_heads),
                self.select_blocks,
                device=coarse_score.device,
                dtype=torch.long,
            )
            positions, _ = self._component_candidate_position_ids(
                layer_idx, query, coarse_score, counts, first
            )
            self.pending_block_ids[layer_idx] = None
            return positions[..., :self.sparse_budget]
        if self.refine_candidate_ratio is None:
            requested_candidates = int(math.ceil(
                self.refine_factor * self.select_blocks
            ))
        else:
            requested_candidates = int(math.ceil(
                self.refine_candidate_ratio * available
            ))
        candidate_count = min(
            available, max(self.select_blocks, requested_candidates)
        )
        candidate_relative = coarse_score.topk(
            candidate_count, dim=-1
        ).indices
        candidate_blocks = candidate_relative + first
        if self.refine_sketch_rank:
            exact_token_logits = self._refine_sketch_logits(
                layer_idx, query, candidate_blocks
            )
        else:
            keys = self._gather_candidate_keys(layer_idx, candidate_blocks)
            exact_token_logits = torch.einsum(
                "bhgqd,bhcsd->bhgqcs", query.float(), keys.float()
            ) / math.sqrt(self.head_dim)
        exact_block_logits = torch.logsumexp(exact_token_logits, dim=-1)

        # Candidate tokens are normalized against the complete approximate
        # pool, with candidate block masses corrected exactly.  Averaging the
        # resulting probabilities selects tokens by total live-query mass
        # across all query heads sharing a KV head.
        hybrid_logits = coarse_logits.clone()
        scatter_index = candidate_relative[:, :, None, None].expand(
            self.batch_size,
            self.num_key_value_heads,
            hybrid_logits.shape[2],
            hybrid_logits.shape[3],
            candidate_count,
        )
        hybrid_logits.scatter_(-1, scatter_index, exact_block_logits)
        log_normalizer = torch.logsumexp(
            hybrid_logits.float(), dim=-1, keepdim=True
        )
        token_probability = (
            exact_token_logits - log_normalizer[..., None]
        ).exp()
        token_score = token_probability.mean(dim=(2, 3)).flatten(-2)
        winner = token_score.topk(self.sparse_budget, dim=-1).indices

        offsets = torch.arange(
            self.block_size,
            device=candidate_blocks.device,
            dtype=candidate_blocks.dtype,
        )
        candidate_positions = (
            candidate_blocks[..., None] * self.block_size + offsets
        ).flatten(-2)
        self.pending_block_ids[layer_idx] = None
        return candidate_positions.gather(-1, winner)
