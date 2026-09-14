"""Symmetric multi-centroid approximation to exact block log-sum-exp."""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations

import torch
from torch import nn

from .kv_cache import ShadowKVCache


@lru_cache(maxsize=None)
def _restricted_growth_partitions(
    block_size: int, n_centroids: int
) -> tuple[tuple[int, ...], ...]:
    """Enumerate every unlabeled partition into exactly ``n_centroids`` sets.

    Restricted-growth strings give each set partition once, unlike arbitrary
    assignments which repeat it ``n_centroids!`` times. The largest case used
    here is S(8,4)=1701, so exact enumeration is inexpensive for diagnostics.
    """
    if not 1 <= n_centroids <= block_size:
        raise ValueError("n_centroids must lie in [1, block_size]")
    labels = [0] * block_size
    result: list[tuple[int, ...]] = []

    def visit(position: int, largest: int) -> None:
        if position == block_size:
            if largest + 1 == n_centroids:
                result.append(tuple(labels))
            return
        remaining_after = block_size - position - 1
        for label in range(min(largest + 1, n_centroids - 1) + 1):
            new_largest = max(largest, label)
            missing = n_centroids - (new_largest + 1)
            if missing > remaining_after:
                continue
            labels[position] = label
            visit(position + 1, new_largest)

    visit(1, 0)
    return tuple(result)


def _squared_distances(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distances, [N,S,D] x [N,R,D] -> [N,S,R]."""
    x2 = x.square().sum(-1, keepdim=True)
    c2 = centers.square().sum(-1).unsqueeze(1)
    return (x2 + c2 - 2.0 * torch.einsum("nsd,nrd->nsr", x, centers)).clamp_min_(0)


@torch.inference_mode()
def fit_minimax_two_centroids(
    keys: torch.Tensor,
    batch_blocks: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly solve centroid-constrained two-ball covering for S <= 8.

    Token zero is fixed in the first side, so the 2^(S-1)-1 masks enumerate
    every unordered non-trivial bipartition exactly once.  Returns the two
    centroids, their counts, the one-ball radius, and the optimal two-ball
    maximum radius.
    """
    block_size, dim = keys.shape[-2:]
    if not 2 <= block_size <= 8:
        raise ValueError("exact minimax solver supports block sizes 2 through 8")
    leading = keys.shape[:-2]
    flat = keys.reshape(-1, block_size, dim)
    device = keys.device

    patterns = torch.arange(
        2 ** (block_size - 1) - 1, device=device, dtype=torch.long
    )
    bit_pos = torch.arange(block_size - 1, device=device)
    tail = ((patterns[:, None] >> bit_pos[None, :]) & 1).bool()
    masks = torch.cat(
        (torch.ones(tail.shape[0], 1, device=device, dtype=torch.bool), tail),
        dim=1,
    )
    mask_f = masks.float()
    count_a = mask_f.sum(-1)
    count_b = float(block_size) - count_a

    all_centers = []
    all_counts = []
    all_r1 = []
    all_r2 = []
    for start in range(0, flat.shape[0], batch_blocks):
        x = flat[start : start + batch_blocks].float()
        total = x.sum(1)
        sum_a = torch.einsum("cs,bsd->bcd", mask_f, x)
        center_a = sum_a / count_a[None, :, None]
        center_b = (total[:, None, :] - sum_a) / count_b[None, :, None]

        x2 = x.square().sum(-1)[:, None, :]
        dist_a = (
            x2
            + center_a.square().sum(-1)[..., None]
            - 2.0 * torch.einsum("bsd,bcd->bcs", x, center_a)
        ).clamp_min_(0)
        dist_b = (
            x2
            + center_b.square().sum(-1)[..., None]
            - 2.0 * torch.einsum("bsd,bcd->bcs", x, center_b)
        ).clamp_min_(0)
        neg_inf = torch.tensor(float("-inf"), device=device)
        radius_a2 = torch.where(
            masks[None], dist_a, neg_inf
        ).amax(-1)
        radius_b2 = torch.where(
            (~masks)[None], dist_b, neg_inf
        ).amax(-1)
        objective = torch.maximum(radius_a2, radius_b2)
        best = objective.argmin(-1)
        rows = torch.arange(x.shape[0], device=device)
        centers = torch.stack(
            (center_a[rows, best], center_b[rows, best]), dim=1
        )
        counts = torch.stack(
            (count_a[best], count_b[best]), dim=1
        ).long()
        mean = x.mean(1)
        r1 = (x - mean[:, None]).square().sum(-1).amax(-1).sqrt()
        r2 = objective[rows, best].sqrt()
        all_centers.append(centers.to(keys.dtype))
        all_counts.append(counts)
        all_r1.append(r1)
        all_r2.append(r2)

    return (
        torch.cat(all_centers).reshape(*leading, 2, dim),
        torch.cat(all_counts).reshape(*leading, 2),
        torch.cat(all_r1).reshape(*leading),
        torch.cat(all_r2).reshape(*leading),
    )


@torch.inference_mode()
def fit_key_only_two_centroids(
    keys: torch.Tensor,
    objective: str,
    batch_blocks: int = 512,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Exactly optimize a query-free two-centroid objective for ``S <= 8``.

    ``scatter`` minimizes normalized within-group Euclidean scatter.  The
    normalization is per block, so the allocation gain compares angular/norm
    dispersion rather than merely preferring heads with a large key scale.
    ``cosine`` minimizes spherical within-group scatter after unit-normalizing
    every key.  In both cases the returned routing centroids remain the raw
    group means, preserving the Jensen lower-bound identity used by the
    centroid-LSE router.
    """
    if objective not in {"scatter", "cosine"}:
        raise ValueError("objective must be scatter or cosine")
    block_size, dim = keys.shape[-2:]
    if not 2 <= block_size <= 8:
        raise ValueError("exact key-only solver supports block sizes 2 through 8")
    leading = keys.shape[:-2]
    flat = keys.reshape(-1, block_size, dim)
    device = keys.device

    patterns = torch.arange(
        2 ** (block_size - 1) - 1, device=device, dtype=torch.long
    )
    bit_pos = torch.arange(block_size - 1, device=device)
    tail = ((patterns[:, None] >> bit_pos[None, :]) & 1).bool()
    masks = torch.cat(
        (torch.ones(tail.shape[0], 1, device=device, dtype=torch.bool), tail),
        dim=1,
    )
    mask_f = masks.float()
    count_a = mask_f.sum(-1)
    count_b = float(block_size) - count_a

    all_centers = []
    all_counts = []
    all_one = []
    all_two = []
    all_gain = []
    for start in range(0, flat.shape[0], batch_blocks):
        x = flat[start : start + batch_blocks].float()
        total = x.sum(1)
        sum_a = torch.einsum("cs,bsd->bcd", mask_f, x)
        sum_b = total[:, None] - sum_a
        center_a = sum_a / count_a[None, :, None]
        center_b = sum_b / count_b[None, :, None]

        if objective == "scatter":
            energy = x.square().sum((1, 2)).clamp_min(1e-12)
            one_raw = energy - total.square().sum(-1) / float(block_size)
            two_raw = (
                energy[:, None]
                - sum_a.square().sum(-1) / count_a[None]
                - sum_b.square().sum(-1) / count_b[None]
            ).clamp_min(0)
            one = one_raw / energy
            candidates = two_raw / energy[:, None]
        else:
            unit = nn.functional.normalize(x, dim=-1, eps=1e-12)
            unit_total = unit.sum(1)
            unit_a = torch.einsum("cs,bsd->bcd", mask_f, unit)
            unit_b = unit_total[:, None] - unit_a
            one = float(block_size) - unit_total.norm(dim=-1)
            candidates = float(block_size) - (
                unit_a.norm(dim=-1) + unit_b.norm(dim=-1)
            )
            one = one / float(block_size)
            candidates = candidates / float(block_size)

        best = candidates.argmin(-1)
        rows = torch.arange(x.shape[0], device=device)
        two = candidates[rows, best]
        centers = torch.stack(
            (center_a[rows, best], center_b[rows, best]), dim=1
        )
        counts = torch.stack((count_a[best], count_b[best]), dim=1).long()
        all_centers.append(centers.to(keys.dtype))
        all_counts.append(counts)
        all_one.append(one)
        all_two.append(two)
        all_gain.append((one - two).clamp_min(0))

    return (
        torch.cat(all_centers).reshape(*leading, 2, dim),
        torch.cat(all_counts).reshape(*leading, 2),
        torch.cat(all_one).reshape(*leading),
        torch.cat(all_two).reshape(*leading),
        torch.cat(all_gain).reshape(*leading),
    )


@torch.inference_mode()
def fit_exact_key_only_r_centroids(
    keys: torch.Tensor,
    n_centroids: int,
    objective: str,
    batch_blocks: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Globally optimize a key-only partition of at most eight tokens.

    The search enumerates every set partition, so the returned solution is a
    global optimum within the family of centroid mixtures. ``scatter`` is
    Euclidean k-means distortion, ``cosine`` is spherical scatter, and
    ``minimax`` is the largest distance from a key to its assigned raw-mean
    centroid. Raw key means and exact populations are always returned; hence
    every resulting centroid mixture is a Jensen lower bound on exact block
    log-sum-exp.
    """
    if objective not in {"scatter", "cosine", "minimax"}:
        raise ValueError("objective must be scatter, cosine, or minimax")
    block_size, dim = keys.shape[-2:]
    if not 1 <= block_size <= 8:
        raise ValueError("exact partition search supports block sizes up to 8")
    assignments = _restricted_growth_partitions(block_size, n_centroids)
    leading = keys.shape[:-2]
    flat = keys.reshape(-1, block_size, dim)
    labels = torch.tensor(assignments, device=keys.device, dtype=torch.long)
    membership = nn.functional.one_hot(
        labels, num_classes=n_centroids
    ).to(torch.float32)
    counts = membership.sum(1)

    all_centers: list[torch.Tensor] = []
    all_counts: list[torch.Tensor] = []
    all_objective: list[torch.Tensor] = []
    for start in range(0, flat.shape[0], batch_blocks):
        x = flat[start : start + batch_blocks].float()
        sums = torch.einsum("psr,bsd->bprd", membership, x)
        centers = sums / counts[None, :, :, None]
        if objective == "scatter":
            energy = x.square().sum((1, 2))
            candidate = energy[:, None] - (
                sums.square().sum(-1) / counts[None]
            ).sum(-1)
            candidate = candidate.clamp_min_(0) / energy[:, None].clamp_min(1e-12)
        elif objective == "cosine":
            unit = nn.functional.normalize(x, dim=-1, eps=1e-12)
            unit_sums = torch.einsum("psr,bsd->bprd", membership, unit)
            candidate = (
                float(block_size) - unit_sums.norm(dim=-1).sum(-1)
            ) / float(block_size)
        else:
            dist2 = (
                x.square().sum(-1)[:, None, :, None]
                + centers.square().sum(-1)[:, :, None, :]
                - 2.0 * torch.einsum("bsd,bprd->bpsr", x, centers)
            ).clamp_min_(0)
            assigned = membership[None].bool()
            candidate = dist2.masked_fill(~assigned, -torch.inf).amax((2, 3)).sqrt()
        best = candidate.argmin(-1)
        rows = torch.arange(x.shape[0], device=x.device)
        all_centers.append(centers[rows, best].to(keys.dtype))
        all_counts.append(counts[best].long())
        all_objective.append(candidate[rows, best])

    return (
        torch.cat(all_centers).reshape(*leading, n_centroids, dim),
        torch.cat(all_counts).reshape(*leading, n_centroids),
        torch.cat(all_objective).reshape(*leading),
    )


@torch.inference_mode()
def fit_self_lse_r_centroids(
    keys: torch.Tensor,
    n_centroids: int,
    temperatures: tuple[float, ...] = (1.0, 1.5, 2.0),
    batch_blocks: int | None = None,
    return_assignment: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Globally optimize an r-centroid partition for normalized-key queries.

    For every key ``k_j`` in a block, the proxy query is
    ``temperature * k_j / ||k_j||``.  A candidate partition is scored by its
    worst Jensen log-sum-exp gap over every proxy direction and temperature.
    All set partitions are enumerated, so this is a diagnostic global optimum
    for blocks of at most eight tokens.  The returned representatives are raw
    key means and exact counts, preserving the Jensen lower-bound property.
    """
    block_size, dim = keys.shape[-2:]
    if not 1 <= block_size <= 8:
        raise ValueError("exact partition search supports block sizes up to 8")
    if not 1 <= n_centroids <= block_size:
        raise ValueError("n_centroids must lie in [1, block_size]")
    if not temperatures or any(value <= 0 for value in temperatures):
        raise ValueError("temperatures must be positive")

    assignments = _restricted_growth_partitions(block_size, n_centroids)
    leading = keys.shape[:-2]
    flat = keys.reshape(-1, block_size, dim)
    labels = torch.tensor(assignments, device=keys.device, dtype=torch.long)
    membership = nn.functional.one_hot(
        labels, num_classes=n_centroids
    ).to(torch.float32)
    counts = membership.sum(1)
    temperature = torch.tensor(
        temperatures, device=keys.device, dtype=torch.float32
    )
    if batch_blocks is None:
        # Keep the dominant [B,T,S,P,R] candidate tensor near 32M FP32
        # elements.  This makes r=2 run in a few large batches while retaining
        # bounded memory for the 1,701-partition r=4 diagnostic.
        elements_per_block = (
            len(temperatures) * block_size * len(assignments) * n_centroids
        )
        batch_blocks = max(1, min(flat.shape[0], 32_000_000 // elements_per_block))

    all_centers: list[torch.Tensor] = []
    all_counts: list[torch.Tensor] = []
    all_risk: list[torch.Tensor] = []
    all_assignment: list[torch.Tensor] = []
    for start in range(0, flat.shape[0], batch_blocks):
        x = flat[start : start + batch_blocks].float()
        unit = nn.functional.normalize(x, dim=-1, eps=1e-12)
        # [block, temperature, proxy_query, key]
        logits = torch.einsum("bjd,bid->bji", unit, x)[:, None]
        logits = logits * temperature[None, :, None, None]
        exact = torch.logsumexp(logits, dim=-1)

        # The proxy-query dot product with a raw group mean can be formed from
        # the 8x8 response matrix; no additional head-dimensional dot products
        # are required for each candidate partition.
        grouped = torch.einsum("btji,pir->btjpr", logits, membership)
        grouped = grouped / counts[None, None, None, :, :]
        lower = torch.logsumexp(
            grouped + counts.log()[None, None, None, :, :], dim=-1
        )
        risk = (exact[..., None] - lower).clamp_min_(0).amax((1, 2))
        best = risk.argmin(-1)
        rows = torch.arange(x.shape[0], device=x.device)

        selected_membership = membership[best]
        selected_counts = counts[best]
        selected_sums = torch.einsum("bsr,bsd->brd", selected_membership, x)
        all_centers.append(
            (selected_sums / selected_counts[..., None]).to(keys.dtype)
        )
        all_counts.append(selected_counts.long())
        all_risk.append(risk[rows, best])
        if return_assignment:
            all_assignment.append(labels[best])

    result = (
        torch.cat(all_centers).reshape(*leading, n_centroids, dim),
        torch.cat(all_counts).reshape(*leading, n_centroids),
        torch.cat(all_risk).reshape(*leading),
    )
    if return_assignment:
        return (*result, torch.cat(all_assignment).reshape(*leading, block_size))
    return result


@torch.inference_mode()
def fit_self_lse_two_centroids(
    keys: torch.Tensor,
    temperatures: tuple[float, ...] = (1.0, 1.5, 2.0),
    return_isotropic_alpha: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
] | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fit self-LSE bipartitions and return the one-to-two risk reduction."""
    fitted = fit_self_lse_r_centroids(
        keys,
        n_centroids=2,
        temperatures=temperatures,
        return_assignment=return_isotropic_alpha,
    )
    if return_isotropic_alpha:
        centers, counts, risk_two, assignment = fitted
    else:
        centers, counts, risk_two = fitted
    block_size = keys.shape[-2]
    x = keys.float()
    unit = nn.functional.normalize(x, dim=-1, eps=1e-12)
    response = torch.einsum("...jd,...id->...ji", unit, x)
    temperature = torch.tensor(
        temperatures, device=keys.device, dtype=torch.float32
    )
    logits = response[..., None, :, :] * temperature[:, None, None]
    exact = torch.logsumexp(logits, dim=-1)
    one = logits.mean(dim=-1) + math.log(float(block_size))
    risk_one = (exact - one).clamp_min_(0).amax(dim=(-2, -1))
    gain = (risk_one - risk_two).clamp_min_(0)
    result = (centers, counts, risk_one, risk_two, gain)
    if return_isotropic_alpha:
        membership = nn.functional.one_hot(assignment, num_classes=2).float()
        residual2 = (
            x[..., :, None, :] - centers.float()[..., None, :, :]
        ).square().sum(-1)
        trace_covariance = (residual2 * membership).sum(-2) / counts.float()
        # Retrieval uses z=q/sqrt(d). Under Sigma ~= tr(Sigma)/d I,
        # 0.5 z^T Sigma z = (||q||^2/d) * tr(Sigma)/(2d).
        isotropic_alpha = trace_covariance / (2.0 * x.shape[-1])
        return (*result, isotropic_alpha)
    return result


@torch.inference_mode()
def _aggregate_self_lse_cost(
    mass_ratio: torch.Tensor,
    exact_log_mass: torch.Tensor,
    cost_mode: str,
    cost_beta: float,
) -> torch.Tensor:
    """Aggregate proxy-query error; leading axis is the flattened block."""
    mass_ratio = mass_ratio.clamp(
        min=torch.finfo(torch.float32).tiny, max=1.0
    )
    gap = (-mass_ratio.log()).clamp_min_(0)
    if cost_mode == "max_gap":
        return gap.amax(dim=(1, 2))
    if cost_mode == "mean_gap":
        # Mean Jensen log-mass gap over the self-K proxy bank.  This is the
        # literal risk-neutral counterpart of ``max_gap``: placement minimizes
        # average log-mass distortion, while a separate allocator may still
        # minimize the largest remaining block risk globally.
        return gap.mean(dim=(1, 2))
    if cost_mode == "mean_relative":
        # Parameter-free rate--distortion objective.  Jensen makes
        # ``mass_ratio = Zhat/Z`` lie in [0,1], so this is exactly the mean
        # fraction of within-block mass missed by the summary on the self-K
        # proxy bank.  Unlike a trimmed/CVaR pair, the same scalar distortion
        # can be used for both center placement and rate allocation.
        return (1.0 - mass_ratio).clamp_min_(0).mean(dim=(1, 2))
    if cost_mode == "trimmed_gap":
        # Robust placement objective: discard the largest ``cost_beta``
        # fraction of proxy-query gaps before averaging.  This is used only
        # to choose a partition; allocation can deliberately use a separate
        # tail-sensitive risk on that fixed path.
        samples = gap.flatten(1, 2)
        keep = max(1, math.ceil((1.0 - cost_beta) * samples.shape[1]))
        return samples.topk(keep, dim=1, largest=False).values.mean(1)
    sample_cost = (
        (1.0 - mass_ratio).clamp_min_(0)
        if "relative" in cost_mode else gap
    )
    samples = sample_cost.flatten(1, 2)
    if cost_mode.startswith("weighted_"):
        log_weight = exact_log_mass.flatten(1, 2)
        log_weight = log_weight - torch.logsumexp(
            log_weight, dim=1, keepdim=True
        )
        while log_weight.ndim < samples.ndim:
            log_weight = log_weight.unsqueeze(-1)
        return torch.logsumexp(
            cost_beta * samples + log_weight, dim=1
        ) / cost_beta
    normalizer = math.log(float(samples.shape[1]))
    return (
        torch.logsumexp(cost_beta * samples, dim=1) - normalizer
    ) / cost_beta


@torch.inference_mode()
def fit_agglomerative_self_lse_paths(
    keys: torch.Tensor,
    temperatures: tuple[float, ...] = (1.0, 1.5, 2.0),
    batch_blocks: int = 4096,
    cost_mode: str = "max_gap",
    cost_beta: float = 4.0,
    proxy_weight: torch.Tensor | None = None,
    robust_tiebreak_epsilon: float | None = None,
    extra_proxy_queries: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a nested 1--S centroid path with only O(S^3) merge trials.

    The path starts from the exact singleton partition.  At every step it
    merges the pair of current components whose merger gives the smallest
    worst Jensen log-sum-exp gap over normalized-key proxy queries.  Thus each
    local coarsening is solved exactly, every finer partition refines the
    preceding one, and the resulting risk curve is non-increasing in the
    number of available components.  For S=8 the complete path evaluates
    28+21+15+10+6+3+1 = 84 candidate mergers per block, fewer than the 127
    bipartitions enumerated by the former two-centroid solver.

    ``max_gap`` is the original minimax Jensen-gap objective.  The smooth
    alternatives aggregate either the Jensen gap itself (``mean_gap`` and
    ``gap_lme``) or the
    *fractional mass missed*, ``1 - exp(L-F)`` (``relative_lme``), with a
    log-mean-exp of inverse temperature ``cost_beta``.  A ``weighted_`` prefix
    gives more influence to self-directions on which the exact block mass is
    large.  These weights use only the block's own keys.

    Returns ``risk[..., r-1]`` and token-to-component assignments
    ``assignment[..., r-1, token]`` for every capacity r in [1,S].
    """
    block_size, dim = keys.shape[-2:]
    if not 1 <= block_size <= 8:
        raise ValueError("agglomerative self-LSE supports blocks up to 8 tokens")
    if not temperatures or any(value <= 0 for value in temperatures):
        raise ValueError("temperatures must be positive")
    if batch_blocks < 1:
        raise ValueError("batch_blocks must be positive")
    smooth_modes = {
        "mean_gap", "gap_lme", "relative_lme",
        "weighted_gap_lme", "weighted_relative_lme",
        "trimmed_gap", "mean_relative", "importance_relative",
    }
    if cost_mode not in {"max_gap", *smooth_modes}:
        raise ValueError(
            "cost_mode must be max_gap, mean_gap, gap_lme, relative_lme, "
            "weighted_gap_lme, weighted_relative_lme, mean_relative, "
            "importance_relative, or trimmed_gap"
        )
    if cost_mode == "importance_relative":
        if proxy_weight is None or proxy_weight.shape != keys.shape[:-1]:
            raise ValueError(
                "importance_relative requires one proxy weight per key"
            )
        if not torch.isfinite(proxy_weight).all() or (proxy_weight < 0).any():
            raise ValueError("proxy weights must be finite and non-negative")
    if robust_tiebreak_epsilon is not None:
        if cost_mode != "importance_relative":
            raise ValueError(
                "robust density tie-break requires importance_relative"
            )
        if robust_tiebreak_epsilon < 0:
            raise ValueError("robust tie-break epsilon must be non-negative")
    if cost_mode == "trimmed_gap" and not 0.0 <= cost_beta < 1.0:
        raise ValueError("trimmed-gap fraction must lie in [0,1)")
    if cost_mode in smooth_modes and cost_mode != "trimmed_gap" and cost_beta <= 0:
        raise ValueError("smooth self-LSE beta must be positive")

    leading = keys.shape[:-2]
    if extra_proxy_queries is not None:
        if (
            extra_proxy_queries.shape[:-2] != leading
            or extra_proxy_queries.shape[-1] != dim
            or extra_proxy_queries.shape[-2] < 1
        ):
            raise ValueError(
                "extra proxy queries must have shape [...,Q,D] matching keys"
            )
    flat = keys.reshape(-1, block_size, dim)
    flat_extra = (
        extra_proxy_queries.reshape(-1, extra_proxy_queries.shape[-2], dim)
        .float()
        if extra_proxy_queries is not None else None
    )
    flat_proxy_weight = (
        proxy_weight.reshape(-1, block_size).float()
        if proxy_weight is not None else None
    )
    temperature = torch.tensor(
        temperatures, device=keys.device, dtype=torch.float32
    )
    all_risk: list[torch.Tensor] = []
    all_assignment: list[torch.Tensor] = []

    for start in range(0, flat.shape[0], batch_blocks):
        x = flat[start : start + batch_blocks].float()
        n_block = x.shape[0]
        unit = nn.functional.normalize(x, dim=-1, eps=1e-12)
        logits = torch.einsum("bjd,bid->bji", unit, x)[:, None]
        if flat_extra is not None:
            anchor_logits = torch.einsum(
                "bqd,bid->bqi", flat_extra[start : start + n_block], x
            )[:, None]
            logits = torch.cat((logits, anchor_logits), dim=2)
        logits = logits * temperature[None, :, None, None]
        exact = torch.logsumexp(logits, dim=-1)

        # At r=S every component is a singleton, so its sufficient statistic
        # in proxy-query space is just the corresponding token logit.
        sum_logits = logits
        counts = torch.ones(
            n_block, block_size, device=x.device, dtype=torch.float32
        )
        labels = torch.arange(
            block_size, device=x.device, dtype=torch.long
        ).expand(n_block, -1).clone()
        risk_path = torch.zeros(
            n_block, block_size, device=x.device, dtype=torch.float32
        )
        assignment_path = torch.empty(
            n_block,
            block_size,
            block_size,
            device=x.device,
            dtype=torch.uint8,
        )
        assignment_path[:, block_size - 1] = labels.to(torch.uint8)

        for r in range(block_size, 1, -1):
            pair_a, pair_b = torch.triu_indices(
                r, r, offset=1, device=x.device
            )
            component = sum_logits / counts[:, None, None, :]
            component = component + counts.log()[:, None, None, :]
            normalized_term = torch.exp(component - exact[..., None])
            normalized_mass = normalized_term.sum(-1)

            merged_count = counts[:, pair_a] + counts[:, pair_b]
            merged_sum = (
                sum_logits[..., pair_a] + sum_logits[..., pair_b]
            )
            merged_component = (
                merged_sum / merged_count[:, None, None, :]
                + merged_count.log()[:, None, None, :]
            )
            candidate_mass = (
                normalized_mass[..., None]
                - normalized_term[..., pair_a]
                - normalized_term[..., pair_b]
                + torch.exp(merged_component - exact[..., None])
            )
            # Jensen guarantees candidate_mass <= 1 in exact arithmetic.
            if cost_mode == "importance_relative":
                weight = flat_proxy_weight[start : start + n_block]
                weight = weight[:, None, :, None]
                missed = (1.0 - candidate_mass).clamp_min(0)
                density_risk = (missed * weight).sum(dim=(1, 2)) / (
                    weight.sum(dim=(1, 2)).clamp_min(1e-12)
                    * float(len(temperatures))
                )
                if robust_tiebreak_epsilon is None:
                    candidate_risk = density_risk
                else:
                    gap = (-candidate_mass.clamp_min(
                        torch.finfo(torch.float32).tiny
                    ).log()).flatten(1, 2)
                    keep = max(1, math.ceil(0.75 * gap.shape[1]))
                    robust_risk = gap.topk(
                        keep, dim=1, largest=False
                    ).values.mean(1)
                    optimum = robust_risk.amin(-1, keepdim=True)
                    feasible = robust_risk <= (
                        (1.0 + robust_tiebreak_epsilon) * optimum + 1e-6
                    )
                    # The selected index is driven by the density objective;
                    # the stored curve uses the primary robust objective.
                    choice_risk = density_risk.masked_fill(
                        ~feasible, float("inf")
                    )
                    best = choice_risk.argmin(-1)
                    rows = torch.arange(n_block, device=x.device)
                    candidate_risk = robust_risk
            else:
                candidate_risk = _aggregate_self_lse_cost(
                    candidate_mass, exact, cost_mode, cost_beta
                )
            if not (
                cost_mode == "importance_relative"
                and robust_tiebreak_epsilon is not None
            ):
                best = candidate_risk.argmin(-1)
                rows = torch.arange(n_block, device=x.device)
            chosen_a = pair_a[best]
            chosen_b = pair_b[best]
            risk_path[:, r - 2] = candidate_risk[rows, best]

            chosen_count = counts[rows, chosen_a] + counts[rows, chosen_b]
            chosen_sum = (
                sum_logits[rows, ..., chosen_a]
                + sum_logits[rows, ..., chosen_b]
            )
            output_index = torch.arange(r - 1, device=x.device)[None]
            old_index = output_index + (output_index >= chosen_b[:, None])
            counts = counts.gather(-1, old_index)
            sum_logits = sum_logits.gather(
                -1,
                old_index[:, None, None, :].expand(
                    -1, sum_logits.shape[1], sum_logits.shape[2], -1
                ),
            )
            counts[rows, chosen_a] = chosen_count
            sum_logits[rows, ..., chosen_a] = chosen_sum

            labels = torch.where(
                labels == chosen_b[:, None], chosen_a[:, None], labels
            )
            labels = torch.where(
                labels > chosen_b[:, None], labels - 1, labels
            )
            assignment_path[:, r - 2] = labels.to(torch.uint8)

        # Numerical noise cannot be allowed to violate the refinement
        # ordering used by the exact resource allocator below.
        risk_path = torch.cummin(risk_path, dim=-1).values
        all_risk.append(risk_path)
        all_assignment.append(assignment_path)

    return (
        torch.cat(all_risk).reshape(*leading, block_size),
        torch.cat(all_assignment).reshape(
            *leading, block_size, block_size
        ),
    )


@torch.inference_mode()
def fit_residual_tail_lse_paths(
    keys: torch.Tensor,
    temperatures: tuple[float, ...] = (1.0,),
    cost_mode: str = "max_gap",
    cost_beta: float = 4.0,
    selection_mode: str = "residual",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nested one-body-plus-exact-tail path for every order 1..S.

    Starting with one body component, each upgrade isolates one key according
    to ``selection_mode``: largest residual from the remaining-body mean,
    least angular coherence with that body, or largest key norm.  The result
    is a nested robust-mean path: r-1 exact tail keys plus one mean for
    everything else.  Its risk is evaluated with the same self-LSE cost as
    the generic partition search, so placement and allocation remain separate.
    """
    block_size, dim = keys.shape[-2:]
    if not 1 <= block_size <= 8:
        raise ValueError("residual-tail path supports blocks up to 8 tokens")
    smooth_modes = {
        "gap_lme", "relative_lme",
        "weighted_gap_lme", "weighted_relative_lme",
        "trimmed_gap",
    }
    if cost_mode not in {"max_gap", *smooth_modes}:
        raise ValueError("unknown residual-tail self-LSE cost")
    if cost_mode == "trimmed_gap" and not 0.0 <= cost_beta < 1.0:
        raise ValueError("trimmed-gap fraction must lie in [0,1)")
    if cost_mode in smooth_modes and cost_mode != "trimmed_gap" and cost_beta <= 0:
        raise ValueError("smooth self-LSE beta must be positive")
    if selection_mode not in {"residual", "angular", "norm"}:
        raise ValueError("tail selection must be residual, angular, or norm")

    leading = keys.shape[:-2]
    flat = keys.reshape(-1, block_size, dim).float()
    n_block = flat.shape[0]
    labels = torch.zeros(
        n_block, block_size, device=keys.device, dtype=torch.long
    )
    active = torch.ones_like(labels, dtype=torch.bool)
    assignment = torch.empty(
        n_block, block_size, block_size,
        device=keys.device, dtype=torch.uint8,
    )
    assignment[:, 0] = labels.to(torch.uint8)
    unit = nn.functional.normalize(flat, dim=-1, eps=1e-12)
    cosine = torch.matmul(unit, unit.transpose(-1, -2))
    key_norm = flat.norm(dim=-1)
    for r in range(2, block_size + 1):
        count = active.sum(-1, keepdim=True).clamp_min(1)
        if selection_mode == "residual":
            body_mean = (
                flat * active[..., None]
            ).sum(-2) / count.to(flat.dtype)
            priority = (flat - body_mean[:, None, :]).square().sum(-1)
        elif selection_mode == "angular":
            # Remove self-similarity and average only over active peers.
            active_float = active.to(cosine.dtype)
            peer_sum = torch.matmul(cosine, active_float[..., None]).squeeze(-1)
            peer_sum = peer_sum - cosine.diagonal(dim1=-2, dim2=-1)
            coherence = peer_sum / (count - 1).clamp_min(1).to(cosine.dtype)
            priority = -coherence
        else:
            priority = key_norm
        isolated = priority.masked_fill(~active, float("-inf")).argmax(-1)
        rows = torch.arange(n_block, device=keys.device)
        labels[rows, isolated] = r - 1
        active[rows, isolated] = False
        assignment[:, r - 1] = labels.to(torch.uint8)

    # At full order every component is a singleton. Canonical labels make
    # component slot j identify token j without storing a redundant index.
    # Earlier orders retain the nested isolation path unchanged.
    assignment[:, -1] = torch.arange(
        block_size, device=keys.device, dtype=torch.uint8
    )

    logits = torch.einsum("bjd,bid->bji", unit, flat)[:, None]
    temperature = torch.tensor(
        temperatures, device=keys.device, dtype=torch.float32
    )
    logits = logits * temperature[None, :, None, None]
    exact = torch.logsumexp(logits, dim=-1)
    risk = torch.empty(
        n_block, block_size, device=keys.device, dtype=torch.float32
    )
    for r in range(1, block_size + 1):
        membership = nn.functional.one_hot(
            assignment[:, r - 1].long(), num_classes=r
        ).float()
        counts = membership.sum(1)
        grouped = torch.einsum("btji,bir->btjr", logits, membership)
        grouped = grouped / counts[:, None, None, :]
        lower = torch.logsumexp(
            grouped + counts.log()[:, None, None, :], dim=-1
        )
        ratio = torch.exp(lower - exact)
        risk[:, r - 1] = _aggregate_self_lse_cost(
            ratio, exact, cost_mode, cost_beta
        )
    risk = torch.cummin(risk, dim=-1).values
    return (
        risk.reshape(*leading, block_size),
        assignment.reshape(*leading, block_size, block_size),
    )


@torch.inference_mode()
def evaluate_self_lse_cvar_path(
    keys: torch.Tensor,
    assignments: torch.Tensor,
    temperatures: tuple[float, ...] = (0.25,),
    tail_fraction: float = 0.25,
    extra_proxy_queries: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a fixed partition path with tail-sensitive Jensen risk.

    Placement and rate allocation have deliberately different objectives.
    ``assignments[..., r-1, :]`` is fitted independently (possibly with a
    robust placement loss).  Here we price that order by empirical CVaR of
    its self-key Jensen gaps, so blocks with a few badly diluted directions
    receive additional centers under marginal water filling.
    """
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0,1]")
    if not temperatures or any(value <= 0 for value in temperatures):
        raise ValueError("temperatures must be positive")
    block_size, dim = keys.shape[-2:]
    leading = keys.shape[:-2]
    if extra_proxy_queries is not None:
        if (
            extra_proxy_queries.shape[:-2] != leading
            or extra_proxy_queries.shape[-1] != dim
            or extra_proxy_queries.shape[-2] < 1
        ):
            raise ValueError(
                "extra proxy queries must have shape [...,Q,D] matching keys"
            )
    x = keys.reshape(-1, block_size, dim).float()
    path = assignments.reshape(-1, block_size, block_size).long()
    unit = nn.functional.normalize(x, dim=-1, eps=1e-12)
    temperature = torch.tensor(
        temperatures, device=x.device, dtype=torch.float32
    )
    logits = torch.einsum("bjd,bid->bji", unit, x)[:, None]
    if extra_proxy_queries is not None:
        extra = extra_proxy_queries.reshape(
            -1, extra_proxy_queries.shape[-2], dim
        ).float()
        logits = torch.cat(
            (logits, torch.einsum("bqd,bid->bqi", extra, x)[:, None]),
            dim=2,
        )
    logits = logits * temperature[None, :, None, None]
    exact = torch.logsumexp(logits, dim=-1)
    tail_count = max(
        1, math.ceil(tail_fraction * logits.shape[1] * logits.shape[2])
    )
    risks = []
    for r in range(1, block_size + 1):
        labels = path[:, r - 1]
        membership = nn.functional.one_hot(
            labels, num_classes=r
        ).float()
        counts = membership.sum(1).clamp_min(1)
        grouped = torch.einsum("btji,bic->btjc", logits, membership)
        grouped = grouped / counts[:, None, None, :]
        lower = torch.logsumexp(
            grouped + counts.log()[:, None, None, :], dim=-1
        )
        gap = (exact - lower).clamp_min(0).flatten(1)
        risks.append(
            gap.topk(tail_count, dim=-1).values.mean(-1)
        )
    risk = torch.stack(risks, dim=-1)
    risk = torch.cummin(risk, dim=-1).values
    return risk.reshape(*leading, block_size)


@torch.inference_mode()
def evaluate_proxy_lse_gap_path(
    keys: torch.Tensor,
    assignments: torch.Tensor,
    proxy_queries: torch.Tensor,
) -> torch.Tensor:
    """Mean Jensen log-mass gap on an explicit causal proxy bank.

    ``proxy_queries`` already contains the attention scale (normally
    ``q_last / sqrt(d)``).  Unlike self-K CVaR this quantity measures exactly
    how much each order loses on the observed prefix query, while retaining a
    complete nested 1..S curve for marginal center allocation.
    """
    block_size, dim = keys.shape[-2:]
    leading = keys.shape[:-2]
    if (
        proxy_queries.shape[:-2] != leading
        or proxy_queries.shape[-1] != dim
        or proxy_queries.shape[-2] < 1
    ):
        raise ValueError(
            "proxy queries must have shape [...,Q,D] matching keys"
        )
    x = keys.reshape(-1, block_size, dim).float()
    path = assignments.reshape(-1, block_size, block_size).long()
    query = proxy_queries.reshape(-1, proxy_queries.shape[-2], dim).float()
    logits = torch.einsum("bqd,bid->bqi", query, x)
    exact = torch.logsumexp(logits, dim=-1)
    risks = []
    for r in range(1, block_size + 1):
        labels = path[:, r - 1]
        membership = nn.functional.one_hot(labels, num_classes=r).float()
        counts = membership.sum(1).clamp_min(1)
        grouped = torch.einsum("bqi,bic->bqc", logits, membership)
        grouped = grouped / counts[:, None, :]
        lower = torch.logsumexp(
            grouped + counts.log()[:, None, :], dim=-1
        )
        risks.append((exact - lower).clamp_min(0).mean(-1))
    risk = torch.stack(risks, dim=-1)
    return torch.cummin(risk, dim=-1).values.reshape(
        *leading, block_size
    )


def _query_conditioned_assignment_families(
    keys: torch.Tensor,
    self_lse_path: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Finite, streaming-friendly block-8 placement hypotheses.

    The observed query selects among hypotheses but never changes the token
    block itself.  Every representation remains a mixture of exact raw-key
    group means with exact populations, so the Jensen property is preserved.
    """
    block_size = keys.shape[-2]
    if block_size != 8:
        raise ValueError("query-conditioned placement currently requires S=8")
    leading = keys.shape[:-2]
    device = keys.device
    token = torch.arange(block_size, device=device)

    def isolation_path(order: torch.Tensor) -> torch.Tensor:
        path = torch.zeros(
            *leading, block_size, block_size,
            device=device, dtype=torch.uint8,
        )
        labels = torch.zeros(
            *leading, block_size, device=device, dtype=torch.long
        )
        for r in range(2, block_size + 1):
            isolated = order[..., r - 2]
            labels.scatter_(-1, isolated[..., None], r - 1)
            path[..., r - 1, :] = labels.to(torch.uint8)
        return path

    norm_order = keys.float().norm(dim=-1).argsort(-1, descending=True)
    unit = nn.functional.normalize(keys.float(), dim=-1, eps=1e-12)
    gram = torch.matmul(unit, unit.transpose(-1, -2))
    eye = torch.eye(block_size, device=device, dtype=torch.bool)
    mean_cos = gram.masked_fill(eye, 0).sum(-1) / float(block_size - 1)
    angular_order = mean_cos.argsort(-1, descending=False)

    contiguous = torch.zeros(
        *leading, block_size, block_size,
        device=device, dtype=torch.uint8,
    )
    for r in range(2, block_size + 1):
        labels = torch.div(token * r, block_size, rounding_mode="floor")
        contiguous[..., r - 1, :] = labels.to(torch.uint8)

    def farthest_path(cosine: bool) -> torch.Tensor:
        x = unit if cosine else keys.float()
        distance = (
            1.0 - torch.matmul(x, x.transpose(-1, -2))
            if cosine else (
                x.square().sum(-1, keepdim=True)
                + x.square().sum(-1).unsqueeze(-2)
                - 2.0 * torch.matmul(x, x.transpose(-1, -2))
            ).clamp_min(0)
        )
        first = (
            (1.0 - mean_cos).argmax(-1)
            if cosine else
            (x - x.mean(-2, keepdim=True)).square().sum(-1).argmax(-1)
        )
        selected = torch.zeros(
            *leading, block_size, device=device, dtype=torch.bool
        )
        selected.scatter_(-1, first[..., None], True)
        nearest = distance.gather(
            -1, first[..., None, None].expand(*leading, block_size, 1)
        ).squeeze(-1)
        path = torch.zeros(
            *leading, block_size, block_size,
            device=device, dtype=torch.uint8,
        )
        seed_ids = [first]
        for r in range(2, block_size + 1):
            next_seed = nearest.masked_fill(selected, -torch.inf).argmax(-1)
            seed_ids.append(next_seed)
            selected.scatter_(-1, next_seed[..., None], True)
            next_distance = distance.gather(
                -1,
                next_seed[..., None, None].expand(*leading, block_size, 1),
            ).squeeze(-1)
            nearest = torch.minimum(nearest, next_distance)
            seeds = torch.stack(seed_ids, dim=-1)
            candidate_distance = distance.gather(
                -1, seeds[..., None, :].expand(*leading, block_size, r)
            )
            path[..., r - 1, :] = candidate_distance.argmin(-1).to(torch.uint8)
        return path

    return (
        self_lse_path,
        isolation_path(norm_order),
        isolation_path(angular_order),
        contiguous,
        farthest_path(True),
        farthest_path(False),
    )


@torch.inference_mode()
def fit_observed_query_lse_paths(
    keys: torch.Tensor,
    query: torch.Tensor,
    self_lse_path: torch.Tensor,
    batch_blocks: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit r=1..8 placement to one already-observed prompt query.

    This is causal: ``query`` is the final query of the available prefix, not
    a future decode query.  It is reshaped into GQA groups and selects the
    finite hypothesis with minimum mean absolute block-LSE error for each
    block and capacity.  The returned relevance is observed block mass and is
    useful for a weighted rate--distortion allocation.
    """
    batch, heads, n_blocks, block_size, dim = keys.shape
    bank = query.float()
    if bank.shape[1] == heads:
        bank = bank[:, :, None]
    elif bank.shape[1] % heads == 0:
        bank = bank.view(
            batch, heads, bank.shape[1] // heads,
            bank.shape[-2], bank.shape[-1],
        )
    else:
        raise ValueError("observed query heads are incompatible with KV heads")
    exact = torch.logsumexp(
        torch.einsum("bhgqd,bhnsd->bhgqns", bank, keys.float())
        / math.sqrt(dim),
        dim=-1,
    )
    relevance = torch.softmax(exact, dim=-1).mean(dim=(2, 3))
    best_risk = torch.full(
        (batch, heads, n_blocks, block_size),
        torch.inf, device=keys.device, dtype=torch.float32,
    )
    best_path = torch.zeros_like(self_lse_path)

    for family_path in _query_conditioned_assignment_families(
        keys, self_lse_path
    ):
        for r in range(1, block_size + 1):
            for start in range(0, n_blocks, batch_blocks):
                stop = min(start + batch_blocks, n_blocks)
                labels = family_path[..., start:stop, r - 1, :].long()
                key = keys[..., start:stop, :, :].float()
                membership = nn.functional.one_hot(
                    labels, num_classes=block_size
                ).float()
                counts = membership.sum(-2)
                sums = torch.einsum(
                    "bhnsr,bhnsd->bhnrd", membership, key
                )
                centers = sums / counts.clamp_min(1)[..., None]
                residual2 = (
                    key[..., :, None, :] - centers[..., None, :, :]
                ).square().sum(-1)
                trace = (
                    residual2 * membership
                ).sum(-2) / counts.clamp_min(1)
                alpha = trace / (2.0 * dim)
                logits = torch.einsum(
                    "bhgqd,bhnrd->bhgqnr", bank, centers
                ) / math.sqrt(dim)
                logits = logits + counts.clamp_min(1).log()[
                    :, :, None, None
                ]
                logits = logits + (
                    bank.square().sum(-1) / dim
                )[..., None, None] * alpha[:, :, None, None]
                logits = logits.masked_fill(
                    (counts == 0)[:, :, None, None], -torch.inf
                )
                approximation = torch.logsumexp(logits, dim=-1)
                risk = (
                    exact[..., start:stop] - approximation
                ).abs().mean(dim=(2, 3))
                current = best_risk[..., start:stop, r - 1]
                improve = risk < current
                best_risk[..., start:stop, r - 1] = torch.where(
                    improve, risk, current
                )
                current_path = best_path[..., start:stop, r - 1, :]
                best_path[..., start:stop, r - 1, :] = torch.where(
                    improve[..., None], labels.to(torch.uint8), current_path
                )
    best_risk[..., -1] = 0
    best_risk = torch.cummin(best_risk, dim=-1).values
    return best_risk, best_path, relevance


@torch.inference_mode()
def fit_observed_query_mass_paths(
    keys: torch.Tensor,
    query: torch.Tensor,
    batch_blocks: int = 256,
    objective: str = "global_mass",
    placement_trim_fraction: float = 0.25,
    tail_fraction: float = 0.25,
    selection_fraction: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a nested 1..S path to exact missed prompt-query mass.

    For an already-observed causal query bank, the distortion of block ``G``
    is

        E_q p_G(q) [1 - Zhat_G(q) / Z_G(q)],

    where ``Zhat`` is the Jensen lower approximation induced by arithmetic
    component means and populations.  Summed over blocks, this is exactly the
    fraction of global attention mass lost by the compressed router.  Hence
    block relevance and local representation error are coupled in one
    calibration-free rate--distortion objective.

    ``topk_hinge`` instead minimizes the one-sided shortlist shortfall

        1{F_G >= theta_K} [theta_K - Fhat_G]_+.

    This has a direct deterministic interpretation for every observed query:
    Jensen gives ``Fhat_G <= F_G``.  Therefore any exact top-K block with
    ``Fhat_G >= theta_K`` outranks every non-top-K block under the approximate
    scores (up to ties).  The hinge spends centers only where that sufficient
    condition is violated, rather than optimizing bulk mass.

    ``topk_hinge_cvar`` uses the same pointwise certificate but prices only
    the largest ``tail_fraction`` of observed-query shortfalls.  At a tail
    containing one query this is a minimax-over-query placement objective,
    which is useful for testing whether rare decode directions rather than
    average shortlist quality cause exact-retrieval failures.

    Starting from singleton components, each step greedily merges the pair
    with minimum resulting distortion.  The returned path is nested and the
    concavified marginal allocator can distribute a global center budget over
    its complete r=1..S curve.
    """
    if objective not in {
        "global_mass", "global_mass_cvar", "topk_hinge",
        "topk_hinge_cvar",
        "trimmed_gap_tail_cvar",
    }:
        raise ValueError("unknown observed-query path objective")
    if objective in {"topk_hinge", "topk_hinge_cvar"}:
        if selection_fraction is None or not 0 < selection_fraction <= 1:
            raise ValueError(
                "topk_hinge requires selection_fraction in (0,1]"
            )
    elif selection_fraction is not None:
        raise ValueError(
            "selection_fraction is only valid for topk_hinge"
        )
    if not 0.0 <= placement_trim_fraction < 1.0:
        raise ValueError("placement trim fraction must lie in [0,1)")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail fraction must lie in (0,1]")

    batch, heads, n_blocks, block_size, dim = keys.shape
    if not 1 <= block_size <= 8:
        raise ValueError("query-mass placement supports blocks up to 8 tokens")
    if batch_blocks < 1:
        raise ValueError("batch_blocks must be positive")

    bank = query.float()
    if bank.ndim == 4:
        if bank.shape[1] == heads:
            bank = bank[:, :, None]
        elif bank.shape[1] % heads == 0:
            bank = bank.view(
                batch, heads, bank.shape[1] // heads,
                bank.shape[-2], bank.shape[-1],
            )
        else:
            raise ValueError("observed query heads are incompatible with KV heads")
    if bank.ndim != 5 or bank.shape[:2] != (batch, heads):
        raise ValueError("query must have shape [B,H,G,Q,D]")

    risk = torch.empty(
        batch, heads, n_blocks, block_size,
        device=keys.device, dtype=torch.float32,
    )
    path = torch.empty(
        batch, heads, n_blocks, block_size, block_size,
        device=keys.device, dtype=torch.uint8,
    )
    relevance = torch.empty(
        batch, heads, n_blocks,
        device=keys.device, dtype=torch.float32,
    )
    sqrt_dim = math.sqrt(dim)

    for batch_idx in range(batch):
        for head in range(heads):
            observed = bank[batch_idx, head].reshape(-1, dim)
            key = keys[batch_idx, head].float()
            exact_all = torch.logsumexp(
                torch.einsum("pd,nsd->nps", observed, key) / sqrt_dim,
                dim=-1,
            )
            probability_all = torch.softmax(exact_all, dim=0)
            relevance[batch_idx, head] = probability_all.mean(-1)
            if objective in {"topk_hinge", "topk_hinge_cvar"}:
                top_blocks = min(
                    n_blocks,
                    max(1, round(float(selection_fraction) * n_blocks)),
                )
                top_mask_all = torch.zeros_like(
                    exact_all, dtype=torch.bool
                )
                top_index = exact_all.topk(top_blocks, dim=0).indices
                top_mask_all.scatter_(0, top_index, True)
                top_threshold = exact_all.topk(
                    top_blocks, dim=0
                ).values[-1]

            for start in range(0, n_blocks, batch_blocks):
                stop = min(start + batch_blocks, n_blocks)
                x = key[start:stop]
                exact = exact_all[start:stop]
                probability = probability_all[start:stop]
                if objective in {"topk_hinge", "topk_hinge_cvar"}:
                    top_mask = top_mask_all[start:stop]
                n_current = x.shape[0]
                sum_logits = (
                    torch.einsum("pd,csd->cps", observed, x) / sqrt_dim
                )
                counts = torch.ones(
                    n_current, block_size,
                    device=keys.device, dtype=torch.float32,
                )
                labels = torch.arange(
                    block_size, device=keys.device, dtype=torch.long,
                ).expand(n_current, -1).clone()
                local_risk = torch.zeros(
                    n_current, block_size,
                    device=keys.device, dtype=torch.float32,
                )
                local_path = torch.empty(
                    n_current, block_size, block_size,
                    device=keys.device, dtype=torch.uint8,
                )
                local_path[:, block_size - 1] = labels.to(torch.uint8)

                for r in range(block_size, 1, -1):
                    pair_a, pair_b = torch.triu_indices(
                        r, r, offset=1, device=keys.device,
                    )
                    component = (
                        sum_logits / counts[:, None, :]
                        + counts.log()[:, None, :]
                    )
                    normalized = torch.exp(component - exact[..., None])
                    normalized_mass = normalized.sum(-1)
                    merged_count = counts[:, pair_a] + counts[:, pair_b]
                    merged_sum = (
                        sum_logits[..., pair_a] + sum_logits[..., pair_b]
                    )
                    merged_component = (
                        merged_sum / merged_count[:, None, :]
                        + merged_count.log()[:, None, :]
                    )
                    candidate_ratio = (
                        normalized_mass[..., None]
                        - normalized[..., pair_a]
                        - normalized[..., pair_b]
                        + torch.exp(merged_component - exact[..., None])
                    ).clamp(max=1.0)
                    if objective in {"global_mass", "global_mass_cvar"}:
                        point_loss = (
                            probability[..., None]
                            * (1.0 - candidate_ratio).clamp_min(0)
                        )
                        if objective == "global_mass":
                            candidate_loss = point_loss.mean(dim=1)
                        else:
                            tail_count = max(
                                1,
                                math.ceil(point_loss.shape[1] * tail_fraction),
                            )
                            candidate_loss = point_loss.topk(
                                tail_count, dim=1
                            ).values.mean(dim=1)
                    elif objective in {"topk_hinge", "topk_hinge_cvar"}:
                        candidate_estimate = exact[..., None] + (
                            candidate_ratio.clamp_min(
                                torch.finfo(torch.float32).tiny
                            ).log()
                        )
                        point_loss = (
                            (
                                top_threshold[None, :, None]
                                - candidate_estimate
                            ).clamp_min(0)
                            * top_mask[..., None]
                        )
                        if objective == "topk_hinge":
                            candidate_loss = point_loss.mean(dim=1)
                        else:
                            tail_count = max(
                                1,
                                math.ceil(
                                    point_loss.shape[1] * tail_fraction
                                ),
                            )
                            candidate_loss = point_loss.topk(
                                tail_count, dim=1
                            ).values.mean(dim=1)
                    else:
                        # This is the query-conditioned counterpart of the
                        # self-K robust placement objective.  Only the proxy
                        # bank changes: candidate partitions still minimize a
                        # trimmed Jensen log-mass gap.  With a single
                        # GQA-mean query the trim is intentionally degenerate,
                        # making this a clean K-proxy -> final-Q control.
                        candidate_gap = -candidate_ratio.clamp_min(
                            torch.finfo(torch.float32).tiny
                        ).log()
                        keep = max(
                            1,
                            math.ceil(
                                (1.0 - placement_trim_fraction)
                                * candidate_gap.shape[1]
                            ),
                        )
                        candidate_loss = candidate_gap.topk(
                            keep, dim=1, largest=False
                        ).values.mean(1)
                    best = candidate_loss.argmin(-1)
                    rows = torch.arange(n_current, device=keys.device)
                    chosen_a = pair_a[best]
                    chosen_b = pair_b[best]
                    if objective in {
                        "global_mass", "global_mass_cvar", "topk_hinge",
                        "topk_hinge_cvar",
                    }:
                        local_risk[:, r - 2] = candidate_loss[rows, best]
                    else:
                        chosen_ratio = candidate_ratio[rows, :, best]
                        chosen_gap = -chosen_ratio.clamp_min(
                            torch.finfo(torch.float32).tiny
                        ).log()
                        tail_count = max(
                            1,
                            math.ceil(tail_fraction * chosen_gap.shape[1]),
                        )
                        local_risk[:, r - 2] = chosen_gap.topk(
                            tail_count, dim=1
                        ).values.mean(1)

                    chosen_count = (
                        counts[rows, chosen_a] + counts[rows, chosen_b]
                    )
                    chosen_sum = (
                        sum_logits[rows, :, chosen_a]
                        + sum_logits[rows, :, chosen_b]
                    )
                    output_index = torch.arange(
                        r - 1, device=keys.device,
                    )[None]
                    old_index = output_index + (
                        output_index >= chosen_b[:, None]
                    )
                    counts = counts.gather(-1, old_index)
                    sum_logits = sum_logits.gather(
                        -1,
                        old_index[:, None, :].expand(
                            -1, observed.shape[0], -1,
                        ),
                    )
                    counts[rows, chosen_a] = chosen_count
                    sum_logits[rows, :, chosen_a] = chosen_sum
                    labels = torch.where(
                        labels == chosen_b[:, None],
                        chosen_a[:, None], labels,
                    )
                    labels = torch.where(
                        labels > chosen_b[:, None], labels - 1, labels,
                    )
                    local_path[:, r - 2] = labels.to(torch.uint8)

                risk[batch_idx, head, start:stop] = torch.cummin(
                    local_risk, dim=-1,
                ).values
                path[batch_idx, head, start:stop] = local_path

    risk[..., -1] = 0
    return torch.cummin(risk, dim=-1).values, path, relevance


@torch.inference_mode()
def fit_observed_query_trimmed_cvar_paths(
    keys: torch.Tensor,
    query: torch.Tensor,
    batch_blocks: int = 256,
    placement_trim_fraction: float = 0.25,
    tail_fraction: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace self-K proxies by an observed final query, and nothing else.

    Placement minimizes the trimmed Jensen gap and allocation prices the
    resulting nested path by tail-CVaR, matching the robust-trimmed/tail-CVaR
    method.  This wrapper exists to keep the experimental control distinct
    from the global-attention-mass objective used by ``qmass_one``.
    """
    return fit_observed_query_mass_paths(
        keys,
        query,
        batch_blocks=batch_blocks,
        objective="trimmed_gap_tail_cvar",
        placement_trim_fraction=placement_trim_fraction,
        tail_fraction=tail_fraction,
    )


@torch.inference_mode()
def fit_ordered_query_lse_paths(
    keys: torch.Tensor,
    query: torch.Tensor,
    batch_blocks: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact contiguous quantization after ordering tokens by observed logit.

    For a single causal query, the relevant geometry is the scalar response
    ``q^T k``.  Tokens are ordered by their mean GQA response, and for every
    capacity r all C(7,r-1) contiguous partitions are enumerated.  The chosen
    partition minimizes mean absolute LSE error across the query heads sharing
    one KV head.  This is a 128-candidate block-8 dynamic program in explicit
    form, not a learned or task-calibrated heuristic.
    """
    batch, heads, n_blocks, block_size, dim = keys.shape
    if block_size != 8:
        raise ValueError("ordered query placement currently requires S=8")
    bank = query.float()
    if bank.shape[1] == heads:
        bank = bank[:, :, None]
    elif bank.shape[1] % heads == 0:
        bank = bank.view(
            batch, heads, bank.shape[1] // heads,
            bank.shape[-2], bank.shape[-1],
        )
    else:
        raise ValueError("observed query heads are incompatible with KV heads")
    # The campaign intentionally uses one observed position.
    bank = bank.mean(dim=-2)
    groups = bank.shape[2]
    exact = torch.logsumexp(
        torch.einsum("bhgd,bhnsd->bhngs", bank, keys.float())
        / math.sqrt(dim),
        dim=-1,
    )
    relevance = torch.softmax(exact.mean(-1), dim=-1)
    flat_key = keys.float().reshape(-1, block_size, dim)
    flat_query = bank[:, :, None].expand(
        batch, heads, n_blocks, groups, dim
    ).reshape(-1, groups, dim)
    flat_exact = exact.reshape(-1, groups)
    output_risk = torch.empty(
        flat_key.shape[0], block_size,
        device=keys.device, dtype=torch.float32,
    )
    output_path = torch.zeros(
        flat_key.shape[0], block_size, block_size,
        device=keys.device, dtype=torch.uint8,
    )

    label_bank = {}
    for r in range(1, block_size + 1):
        candidates = []
        for cuts in combinations(range(1, block_size), r - 1):
            boundaries = (0, *cuts, block_size)
            labels = torch.empty(block_size, dtype=torch.long)
            for group, (left, right) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            ):
                labels[left:right] = group
            candidates.append(labels)
        label_bank[r] = torch.stack(candidates).to(keys.device)

    for start in range(0, flat_key.shape[0], batch_blocks):
        stop = min(start + batch_blocks, flat_key.shape[0])
        key = flat_key[start:stop]
        q = flat_query[start:stop]
        exact_b = flat_exact[start:stop]
        token_logits = torch.einsum("bgd,bsd->bgs", q, key) / math.sqrt(dim)
        order = token_logits.mean(1).argsort(-1)
        qnorm2 = q.square().sum(-1) / dim
        key_energy = key.square().sum(-1)
        for r in range(1, block_size + 1):
            sorted_labels = label_bank[r]
            n_candidate = sorted_labels.shape[0]
            labels = torch.empty(
                key.shape[0], n_candidate, block_size,
                device=keys.device, dtype=torch.long,
            )
            labels.scatter_(
                -1,
                order[:, None].expand(-1, n_candidate, -1),
                sorted_labels[None].expand(key.shape[0], -1, -1),
            )
            membership = nn.functional.one_hot(
                labels, num_classes=r
            ).float()
            counts = membership.sum(-2)
            grouped_logits = torch.einsum(
                "bgs,bpsr->bgpr", token_logits, membership
            ) / counts[:, None]
            sums = torch.einsum("bpsr,bsd->bprd", membership, key)
            centers = sums / counts[..., None]
            energy = torch.einsum("bpsr,bs->bpr", membership, key_energy)
            trace = (
                energy / counts - centers.square().sum(-1)
            ).clamp_min(0)
            alpha = trace / (2.0 * dim)
            component = grouped_logits + counts.log()[:, None]
            component = component + qnorm2[:, :, None, None] * alpha[:, None]
            approximation = torch.logsumexp(component, dim=-1)
            risk = (flat_exact[start:stop, :, None] - approximation).abs().mean(1)
            best = risk.argmin(-1)
            rows = torch.arange(key.shape[0], device=keys.device)
            output_risk[start:stop, r - 1] = risk[rows, best]
            output_path[start:stop, r - 1] = labels[rows, best].to(torch.uint8)

    output_risk[..., -1] = 0
    output_risk = torch.cummin(output_risk, dim=-1).values
    return (
        output_risk.reshape(batch, heads, n_blocks, block_size),
        output_path.reshape(batch, heads, n_blocks, block_size, block_size),
        relevance,
    )


@torch.inference_mode()
def allocate_concave_marginal_counts(
    risk: torch.Tensor,
    extra_fraction: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fixed-quota water filling on a concavified distortion curve."""
    n_blocks, stages = risk.shape[-2:]
    n_extra = int(round(extra_fraction * n_blocks))
    gain = (risk[..., :-1] - risk[..., 1:]).clamp_min(0)
    if weight is not None:
        gain = gain * weight[..., None]
    gain = torch.cummin(gain, dim=-1).values
    flat = gain.flatten(-2)
    n_extra = min(n_extra, flat.shape[-1])
    counts = torch.ones_like(risk[..., 0], dtype=torch.long)
    if n_extra:
        chosen = flat.topk(n_extra, dim=-1).indices
        selected = torch.zeros_like(flat, dtype=torch.bool)
        selected.scatter_(-1, chosen, True)
        counts += selected.view(*risk.shape[:-2], n_blocks, stages - 1).sum(-1)
    return counts


@torch.inference_mode()
def priced_concave_marginal_counts(
    risk: torch.Tensor,
    penalty: torch.Tensor,
) -> torch.Tensor:
    """Buy every sequential center whose concavified gain clears a price.

    This is the quota-free counterpart of
    :func:`allocate_concave_marginal_counts`.  The realized center count is an
    output.  ``cummin`` enforces precedence and diminishing marginal returns.
    """
    gain = (risk[..., :-1] - risk[..., 1:]).clamp_min(0)
    gain = torch.cummin(gain, dim=-1).values
    return 1 + (gain >= penalty[..., None, None]).sum(dim=-1)


@torch.inference_mode()
def distortion_threshold_counts(
    risk: torch.Tensor,
    tolerance: torch.Tensor,
) -> torch.Tensor:
    """Use the smallest representation order meeting an absolute tolerance."""
    feasible = risk <= tolerance[..., None, None]
    # The final singleton partition has zero error, hence every path is
    # feasible by its last stage and argmax returns the first feasible order.
    return feasible.to(torch.uint8).argmax(dim=-1).long() + 1


@torch.inference_mode()
def rate_distortion_counts(
    risk: torch.Tensor,
    penalty: torch.Tensor,
) -> torch.Tensor:
    """Choose every block order independently for a Lagrange penalty.

    ``risk[..., block, r-1]`` is the distortion at representation order r.
    The returned order minimizes ``risk + penalty * (r-1)``.  The operation
    is one vectorized argmin over the complete 1..S path; it does not assign
    centers one at a time.
    """
    rates = torch.arange(
        risk.shape[-1], device=risk.device, dtype=risk.dtype
    )
    objective = risk + penalty[..., None, None] * rates
    return objective.argmin(dim=-1) + 1


@torch.inference_mode()
def relative_rate_distortion_counts(
    risk: torch.Tensor,
    penalty: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Choose every block order without imposing a global center quota.

    Normalizing each path by its one-center distortion makes the shared rate
    penalty dimensionless and invariant to layer/head logit scale.  Blocks
    with numerically zero one-center distortion remain at order one.
    """
    base = risk[..., :1]
    normalized = torch.where(
        base > eps,
        risk / base.clamp_min(eps),
        torch.zeros_like(risk),
    )
    return rate_distortion_counts(normalized, penalty)


@torch.inference_mode()
def allocate_rate_distortion_counts(
    risk: torch.Tensor,
    extra_fraction: float,
    iterations: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the single rate penalty whose solution respects a center cap.

    This solves the Lagrangian relaxation

        min sum_G D_G(r_G) + lambda * sum_G (r_G - 1)

    in parallel for all blocks.  A batched bisection selects the smallest
    penalty whose independently optimal orders use no more than the requested
    mean number of centers.  Discrete ties may leave a few centers unused;
    they are deliberately not filled by a second, unrelated allocator.
    """
    n_blocks, stages = risk.shape[-2:]
    if not 0.0 <= extra_fraction <= float(stages - 1):
        raise ValueError(
            f"extra_fraction must lie in [0,{stages - 1}]"
        )
    target = n_blocks + int(round(extra_fraction * n_blocks))
    prefix = risk.shape[:-2]
    low = torch.zeros(prefix, device=risk.device, dtype=risk.dtype)
    # Any extra center can reduce distortion by at most D_G(1).  A penalty
    # above the largest one-center distortion therefore selects r=1.
    high = risk[..., 0].amax(dim=-1).clamp_min(0) + 1e-6
    zero_counts = rate_distortion_counts(risk, low)
    if torch.all(zero_counts.sum(-1) <= target):
        return zero_counts, low
    for _ in range(iterations):
        mid = (low + high) * 0.5
        total = rate_distortion_counts(risk, mid).sum(-1)
        too_many = total > target
        low = torch.where(too_many, mid, low)
        high = torch.where(too_many, high, mid)
    counts = rate_distortion_counts(risk, high)
    return counts, high


@torch.inference_mode()
def allocate_distortion_target_counts(
    risk: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimize representation rate subject to a mean-distortion target.

    The target is imposed independently for every leading group (in the
    streaming cache this is one batch/layer/KV-head).  A group whose
    one-center representation already meets the target stays entirely at
    order one.  Harder groups automatically receive a lower Lagrange price
    and therefore more centers.  No task label or fixed center quota enters
    the decision.

    The discrete multiple-choice problem is solved on its Lagrangian
    frontier.  Bisection returns the largest feasible center price, i.e. the
    lowest-rate supported solution whose selected mean risk is at most the
    requested target.
    """
    if risk.ndim < 2:
        raise ValueError("risk must end in [blocks, stages]")
    if torch.any(target < 0):
        raise ValueError("distortion target must be non-negative")

    target = target.to(device=risk.device, dtype=risk.dtype)
    prefix = risk.shape[:-2]
    if target.shape != prefix:
        target = torch.broadcast_to(target, prefix)

    low = torch.zeros(prefix, device=risk.device, dtype=risk.dtype)
    high = risk[..., 0].amax(dim=-1).clamp_min(0) + 1e-6

    def selected_mean(price: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts = rate_distortion_counts(risk, price)
        selected = risk.gather(
            -1, (counts - 1).unsqueeze(-1)
        ).squeeze(-1)
        return counts, selected.mean(dim=-1)

    one = torch.ones(risk.shape[:-1], device=risk.device, dtype=torch.long)
    one_mean = risk[..., 0].mean(dim=-1)
    already_feasible = one_mean <= target

    # price=0 is the minimum-risk endpoint and is feasible whenever the
    # requested target can be represented by the available 1..S path.
    _, minimum_mean = selected_mean(low)
    reachable = minimum_mean <= target
    for _ in range(iterations):
        mid = (low + high) * 0.5
        _, distortion = selected_mean(mid)
        feasible = distortion <= target
        low = torch.where(feasible, mid, low)
        high = torch.where(feasible, high, mid)

    counts, _ = selected_mean(low)
    counts = torch.where(already_feasible[..., None], one, counts)
    counts = torch.where(reachable[..., None], counts, rate_distortion_counts(risk, torch.zeros_like(low)))
    price = torch.where(already_feasible, high, low)
    price = torch.where(reachable, price, torch.zeros_like(price))
    return counts, price


@torch.inference_mode()
def demand_adaptive_rate_distortion_counts(
    risk: torch.Tensor,
    low_penalty: float,
    high_penalty: float,
    demand_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raise the center price only for layers already cheap at a safe price.

    ``risk`` ends in ``[heads, blocks, stages]`` (with optional leading batch
    dimensions).  Demand is the mean selected order at ``low_penalty``, pooled
    over the layer's heads and blocks.  Layers above ``demand_threshold`` keep
    the conservative price; saturated layers switch to ``high_penalty``.

    This is deliberately one-sided: a difficult layer is never made cheaper
    than the validated conservative operating point.  The rule removes cost
    only where the representation itself reports low demand, without a task
    label or a fixed global center budget.
    """
    if risk.ndim < 3:
        raise ValueError("risk must end in [heads, blocks, stages]")
    if not 0 <= low_penalty <= high_penalty:
        raise ValueError("penalties must satisfy 0 <= low <= high")
    if demand_threshold < 1:
        raise ValueError("demand threshold must be at least one center")

    low = torch.full(
        risk.shape[:-2], low_penalty, device=risk.device, dtype=risk.dtype
    )
    low_counts = rate_distortion_counts(risk, low)
    # Pool over heads and blocks, retaining any leading batch dimensions.
    demand = low_counts.float().mean(dim=(-2, -1))
    use_high = demand <= demand_threshold
    layer_price = torch.where(
        use_high,
        torch.full_like(demand, high_penalty),
        torch.full_like(demand, low_penalty),
    )
    price = layer_price[..., None].expand(risk.shape[:-2])
    counts = rate_distortion_counts(risk, price)
    return counts, price, demand


@torch.inference_mode()
def hierarchical_log_rate_distortion_counts(
    risk: torch.Tensor,
    alpha: float,
    beta: float = 1.0,
    group_mode: str = "head",
    iterations: int = 12,
    base_penalty: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Optimize a smooth hierarchical rate--distortion objective.

    For each head (or whole layer), this minimizes

    .. math::

       \overline{R_G(r_G)} + \lambda_0\overline{r_G-1}
       + \alpha\log(\beta + \overline{r_G-1}).

    Integrating a shared exponential center price under a Gamma hyper-prior
    gives this log complexity penalty.  Its marginal price
    ``base_penalty + alpha / (beta + mean_extra)`` decreases smoothly only
    when a coherent group contains many useful upgrades.  The non-zero base
    is a lower bound on the center price, preventing a difficult group from
    making late centers artificially free.  Easy heads therefore remain near
    one center, whereas genuinely difficult heads may spend more, without a
    task label, a hard demand threshold, or a fixed quota.

    The concave log term is optimized by majorization--minimization.  Its
    tangent produces one vectorized priced rate--distortion update.  Because
    a concave objective can have two boundary basins, we run MM from the
    one-center and full-resolution endpoints and return the lower exact
    objective.  Runtime is independent of the realized number of centers.
    """
    if risk.ndim < 3:
        raise ValueError("risk must end in [heads, blocks, stages]")
    if alpha < 0 or beta <= 0 or base_penalty < 0:
        raise ValueError(
            "alpha and base_penalty must be non-negative and beta positive"
        )
    if group_mode not in {"head", "layer"}:
        raise ValueError("group_mode must be head or layer")

    stages = risk.shape[-1]

    def expand_price(group_price: torch.Tensor) -> torch.Tensor:
        if group_mode == "head":
            return group_price
        return group_price[..., None].expand(risk.shape[:-2])

    def complexity(counts: torch.Tensor) -> torch.Tensor:
        extra = counts.float() - 1.0
        if group_mode == "head":
            return extra.mean(dim=-1)
        return extra.mean(dim=(-2, -1))

    def solve(initial: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = risk.shape[:-2] if group_mode == "head" else risk.shape[:-3]
        mean_extra = torch.full(
            shape, initial, device=risk.device, dtype=risk.dtype
        )
        counts = torch.ones(risk.shape[:-1], device=risk.device, dtype=torch.long)
        for _ in range(iterations):
            group_price = base_penalty + alpha / (beta + mean_extra)
            counts = rate_distortion_counts(risk, expand_price(group_price))
            mean_extra = complexity(counts).to(risk.dtype)
        group_price = base_penalty + alpha / (beta + mean_extra)
        counts = rate_distortion_counts(risk, expand_price(group_price))
        mean_extra = complexity(counts).to(risk.dtype)
        return counts, expand_price(group_price), mean_extra

    low_counts, low_price, low_extra = solve(0.0)
    high_counts, high_price, high_extra = solve(float(stages - 1))

    def objective(counts: torch.Tensor, mean_extra: torch.Tensor) -> torch.Tensor:
        selected = risk.gather(
            -1, (counts - 1).unsqueeze(-1)
        ).squeeze(-1)
        if group_mode == "head":
            distortion = selected.mean(dim=-1)
        else:
            distortion = selected.mean(dim=(-2, -1))
        return (
            distortion
            + base_penalty * mean_extra
            + alpha * torch.log(beta + mean_extra)
        )

    choose_low = objective(low_counts, low_extra) <= objective(
        high_counts, high_extra
    )
    count_mask = (
        choose_low[..., None]
        if group_mode == "head"
        else choose_low[..., None, None]
    )
    price_mask = (
        choose_low if group_mode == "head" else choose_low[..., None]
    )
    counts = torch.where(count_mask, low_counts, high_counts)
    price = torch.where(price_mask, low_price, high_price)
    mean_extra = torch.where(choose_low, low_extra, high_extra)
    return counts, price, mean_extra


@torch.inference_mode()
def allocate_minimax_centroid_counts(
    risk_path: torch.Tensor,
    extra_fraction: float,
) -> torch.Tensor:
    """Exactly solve the discrete minimax allocation on nested risk paths.

    ``risk_path[..., block, r-1]`` must be non-increasing in r.  Starting from
    one component per block, an upgrade r->r+1 removes the current risk level
    R(block,r).  Selecting the K largest such levels therefore minimizes the
    maximum remaining risk under exactly K extra components.  Monotonicity
    makes the prerequisite constraint automatic; stable sorting resolves ties
    in favor of the earlier upgrade of the same block.
    """
    n_blocks, max_components = risk_path.shape[-2:]
    if max_components < 1:
        raise ValueError("risk_path needs at least one component capacity")
    max_extra = float(max_components - 1)
    if not 0.0 <= extra_fraction <= max_extra:
        raise ValueError(
            f"extra_fraction must lie in [0,{max_extra:g}]"
        )
    if torch.any(risk_path[..., 1:] > risk_path[..., :-1] + 2e-5):
        raise ValueError("risk paths must be non-increasing")
    n_extra = int(round(extra_fraction * n_blocks))
    counts = torch.ones(
        risk_path.shape[:-1], device=risk_path.device, dtype=torch.long
    )
    if not n_extra or max_components == 1:
        return counts

    priority = risk_path[..., :-1].reshape(
        *risk_path.shape[:-2], n_blocks * (max_components - 1)
    )
    # The flattening order is block-major and stage-minor, so stable sorting
    # puts r->r+1 before later equal-risk upgrades of the same block.
    order = torch.argsort(priority, dim=-1, descending=True, stable=True)
    chosen = order[..., :n_extra]
    selected = torch.zeros_like(priority, dtype=torch.bool)
    selected.scatter_(dim=-1, index=chosen, value=True)
    selected = selected.view(
        *risk_path.shape[:-2], n_blocks, max_components - 1
    )
    counts = counts + selected.sum(-1)
    return counts


@torch.inference_mode()
def fit_lazy_exact_self_lse_allocation(
    keys: torch.Tensor,
    extra_fraction: float,
    temperatures: tuple[float, ...] = (1.0, 1.5, 2.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve variable-order self-LSE minimax allocation lazily and exactly.

    Let R_G(r) be the globally minimal worst proxy-query Jensen gap attainable
    by an r-component partition of block G.  The constrained problem is

        min max_G R_G(r_G),  sum_G (r_G - 1) <= K.

    The next upgrade of a block currently at r has priority R_G(r).  Starting
    with exact R(1) and R(2), we select the K largest known priorities.  Exact
    R(3) is evaluated only for blocks that reach three components; it is then
    inserted as the priority of their next upgrade, and allocation is solved
    again.  Repeating this procedure is exact because R_G is non-increasing:
    an unknown later transition can never outrank its unselected prerequisite.

    For S=8 and K<=number_of_blocks, the common case evaluates all 127
    bipartitions once and only a tiny selected tail at r>=3.
    """
    if not 0.0 <= extra_fraction <= 1.0:
        raise ValueError("extra_fraction must lie in [0,1]")
    n_blocks, block_size, dim = keys.shape[-3:]
    if not 1 <= block_size <= 8:
        raise ValueError("lazy exact self-LSE supports blocks up to 8 tokens")
    if not temperatures or any(value <= 0 for value in temperatures):
        raise ValueError("temperatures must be positive")

    leading = keys.shape[:-3]
    x = keys.float()
    unit = nn.functional.normalize(x, dim=-1, eps=1e-12)
    response = torch.einsum("...njd,...nid->...nji", unit, x)
    temperature = torch.tensor(
        temperatures, device=keys.device, dtype=torch.float32
    )
    logits = response.unsqueeze(-3) * temperature.view(
        *([1] * (response.ndim - 2)), -1, 1, 1
    )
    exact = torch.logsumexp(logits, dim=-1)
    one = logits.mean(dim=-1) + math.log(float(block_size))
    risk_one = (exact - one).clamp_min_(0).amax(dim=(-2, -1))

    assignment_path = torch.zeros(
        *leading,
        n_blocks,
        block_size,
        block_size,
        device=keys.device,
        dtype=torch.uint8,
    )
    risk_path = torch.full(
        (*leading, n_blocks, block_size),
        float("nan"),
        device=keys.device,
        dtype=torch.float32,
    )
    risk_path[..., 0] = risk_one
    if block_size == 1:
        return risk_path, assignment_path, torch.ones_like(risk_one, dtype=torch.long)

    _, _, risk_two, assignment_two = fit_self_lse_r_centroids(
        keys,
        n_centroids=2,
        temperatures=temperatures,
        return_assignment=True,
    )
    risk_two = torch.minimum(risk_two, risk_one)
    risk_path[..., 1] = risk_two
    assignment_path[..., 1, :] = assignment_two.to(torch.uint8)
    if block_size == 2:
        assignment_path[..., 1, :] = torch.arange(
            2, device=keys.device, dtype=torch.uint8
        )

    n_extra = int(round(extra_fraction * n_blocks))
    priorities = torch.full(
        (*leading, n_blocks, block_size - 1),
        float("-inf"),
        device=keys.device,
        dtype=torch.float32,
    )
    priorities[..., 0] = risk_one
    if block_size > 2:
        priorities[..., 1] = risk_two

    component_counts = torch.ones_like(risk_one, dtype=torch.long)
    for _ in range(block_size):
        if n_extra:
            flat_priority = priorities.reshape(
                *leading, n_blocks * (block_size - 1)
            )
            order = torch.argsort(
                flat_priority, dim=-1, descending=True, stable=True
            )
            selected = torch.zeros_like(flat_priority, dtype=torch.bool)
            selected.scatter_(
                -1, order[..., :n_extra], torch.ones_like(order[..., :n_extra], dtype=torch.bool)
            )
            component_counts = 1 + selected.view_as(priorities).sum(-1)
        else:
            component_counts.fill_(1)

        discovered = False
        max_allocated = int(component_counts.max().item())
        for r in range(3, max_allocated + 1):
            need = (component_counts >= r) & torch.isnan(risk_path[..., r - 1])
            if not need.any():
                continue
            _, _, selected_risk, selected_assignment = fit_self_lse_r_centroids(
                keys[need],
                n_centroids=r,
                temperatures=temperatures,
                return_assignment=True,
            )
            previous = risk_path[..., r - 2][need]
            selected_risk = torch.minimum(selected_risk, previous)
            risk_path[..., r - 1][need] = selected_risk
            assignment_path[..., r - 1, :][need] = selected_assignment.to(
                torch.uint8
            )
            if r < block_size:
                priorities[..., r - 1][need] = selected_risk
            discovered = True
        if not discovered:
            break
    else:
        raise RuntimeError("lazy exact allocation did not converge")

    if torch.any(component_counts == block_size):
        identity = torch.arange(
            block_size, device=keys.device, dtype=torch.uint8
        )
        assignment_path[..., block_size - 1, :][
            component_counts == block_size
        ] = identity
        risk_path[..., block_size - 1][component_counts == block_size] = 0.0
    return risk_path, assignment_path, component_counts


@torch.inference_mode()
def fit_angular_isolation_lse_paths(
    keys: torch.Tensor,
    temperatures: tuple[float, ...] = (1.0,),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the query-free one-to-two angular-isolation path.

    The second component isolates the key with the smallest mean cosine to
    the other keys in its sealed block.  This is the token whose directional
    peak is diluted most strongly by a single block mean.  Allocation is kept
    separate: ``risk[..., 0] - risk[..., 1]`` is the self-key LSE error
    removed by buying that component.  No prompt or future query is observed.

    This path is deliberately limited to two components.  It is the cheap
    production form of the finite-family diagnostic, and supports the paper's
    1.25/1.5 average-component operating points without enumerating all 127
    bipartitions of an eight-token block.
    """
    if not temperatures or any(value <= 0 for value in temperatures):
        raise ValueError("temperatures must be positive")
    block_size, dim = keys.shape[-2:]
    if block_size < 2:
        raise ValueError("angular isolation requires at least two tokens")

    x = keys.float()
    unit = nn.functional.normalize(x, dim=-1, eps=1e-12)
    gram = torch.matmul(unit, unit.transpose(-1, -2))
    eye = torch.eye(block_size, device=keys.device, dtype=torch.bool)
    mean_cos = gram.masked_fill(eye, 0).sum(-1) / float(block_size - 1)
    isolated = mean_cos.argmin(-1)

    leading = keys.shape[:-2]
    assignment = torch.zeros(
        *leading, 2, block_size, device=keys.device, dtype=torch.uint8
    )
    assignment[..., 1, :].scatter_(
        -1, isolated[..., None], torch.ones_like(isolated[..., None], dtype=torch.uint8)
    )

    temperature = torch.tensor(
        temperatures, device=keys.device, dtype=torch.float32
    )
    proxy_logits = torch.einsum("...jd,...id->...ji", unit, x)
    exact_logits = proxy_logits.unsqueeze(-3) * temperature.view(
        *([1] * (proxy_logits.ndim - 2)), -1, 1, 1
    )
    exact_lse = torch.logsumexp(exact_logits, dim=-1)

    risks = []
    for rank in range(2):
        labels = assignment[..., rank, :].long()
        membership = nn.functional.one_hot(labels, num_classes=2).float()
        counts = membership.sum(-2)
        centers = torch.einsum(
            "...sr,...sd->...rd", membership, x
        ) / counts.clamp_min(1)[..., None]
        residual2 = (x[..., :, None, :] - centers[..., None, :, :]).square().sum(-1)
        trace = (residual2 * membership).sum(-2) / counts.clamp_min(1)
        alpha = trace / (2.0 * float(dim))
        component = torch.einsum("...jd,...rd->...jr", unit, centers)
        component = component.unsqueeze(-3) * temperature.view(
            *([1] * (component.ndim - 2)), -1, 1, 1
        )
        component = component + counts.clamp_min(1).log().unsqueeze(-2).unsqueeze(-2)
        component = component + alpha.unsqueeze(-2).unsqueeze(-2) * temperature.square().view(
            *([1] * (component.ndim - 3)), -1, 1, 1
        )
        component = component.masked_fill(
            (counts == 0).unsqueeze(-2).unsqueeze(-2), -torch.inf
        )
        approximate = torch.logsumexp(component, dim=-1)
        risks.append((exact_lse - approximate).abs().mean(dim=(-2, -1)))

    risk = torch.stack(risks, dim=-1)
    risk[..., 1] = torch.minimum(risk[..., 0], risk[..., 1])
    return risk, assignment


@torch.inference_mode()
def fit_residual_tail_path(
    keys: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a one-to-two path that isolates the largest K residual.

    The one-center representation is the block mean.  The two-center
    representation stores the key farthest from that mean as a singleton and
    represents the other keys by their exact arithmetic mean.  ``priority``
    is the one-center covering radius and is used to allocate the scarce
    second components.  The rule observes sealed K only and costs O(SD).
    """
    block_size = keys.shape[-2]
    if block_size < 2:
        raise ValueError("residual-tail placement requires at least two tokens")
    x = keys.float()
    mean = x.mean(-2, keepdim=True)
    residual = (x - mean).norm(dim=-1)
    isolated = residual.argmax(-1)

    leading = keys.shape[:-2]
    assignment = torch.zeros(
        *leading, 2, block_size, device=keys.device, dtype=torch.uint8
    )
    assignment[..., 1, :].fill_(1)
    assignment[..., 1, :].scatter_(
        -1,
        isolated[..., None],
        torch.zeros_like(isolated[..., None], dtype=torch.uint8),
    )
    return residual.amax(-1), assignment


@torch.inference_mode()
def fit_residual_isolation_lse_paths(
    keys: torch.Tensor,
    temperatures: tuple[float, ...] = (1.0,),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isolate the largest K residual and score its self-LSE benefit.

    Placement remains the O(SD), query-free residual rule.  Unlike
    :func:`fit_residual_tail_path`, allocation uses the actual reduction in
    the self-key log-sum-exp approximation, keeping placement and rate
    allocation aligned with the router objective.
    """
    if not temperatures or any(value <= 0 for value in temperatures):
        raise ValueError("temperatures must be positive")
    block_size, dim = keys.shape[-2:]
    if block_size < 2:
        raise ValueError("residual isolation requires at least two tokens")

    x = keys.float()
    unit = nn.functional.normalize(x, dim=-1, eps=1e-12)
    isolated = (x - x.mean(-2, keepdim=True)).norm(dim=-1).argmax(-1)
    leading = keys.shape[:-2]
    assignment = torch.zeros(
        *leading, 2, block_size, device=keys.device, dtype=torch.uint8
    )
    assignment[..., 1, :].fill_(1)
    assignment[..., 1, :].scatter_(
        -1,
        isolated[..., None],
        torch.zeros_like(isolated[..., None], dtype=torch.uint8),
    )

    temperature = torch.tensor(
        temperatures, device=keys.device, dtype=torch.float32
    )
    proxy_logits = torch.einsum("...jd,...id->...ji", unit, x)
    exact_logits = proxy_logits.unsqueeze(-3) * temperature.view(
        *([1] * (proxy_logits.ndim - 2)), -1, 1, 1
    )
    exact_lse = torch.logsumexp(exact_logits, dim=-1)

    risks = []
    for rank in range(2):
        labels = assignment[..., rank, :].long()
        membership = nn.functional.one_hot(labels, num_classes=2).float()
        counts = membership.sum(-2)
        centers = torch.einsum(
            "...sr,...sd->...rd", membership, x
        ) / counts.clamp_min(1)[..., None]
        residual2 = (
            x[..., :, None, :] - centers[..., None, :, :]
        ).square().sum(-1)
        trace = (residual2 * membership).sum(-2) / counts.clamp_min(1)
        alpha = trace / (2.0 * float(dim))
        component = torch.einsum("...jd,...rd->...jr", unit, centers)
        component = component.unsqueeze(-3) * temperature.view(
            *([1] * (component.ndim - 2)), -1, 1, 1
        )
        component = (
            component
            + counts.clamp_min(1).log().unsqueeze(-2).unsqueeze(-2)
            + alpha.unsqueeze(-2).unsqueeze(-2)
            * temperature.square().view(
                *([1] * (component.ndim - 3)), -1, 1, 1
            )
        )
        component = component.masked_fill(
            (counts == 0).unsqueeze(-2).unsqueeze(-2), -torch.inf
        )
        approximate = torch.logsumexp(component, dim=-1)
        risks.append((exact_lse - approximate).abs().mean(dim=(-2, -1)))

    risk = torch.stack(risks, dim=-1)
    risk[..., 1] = torch.minimum(risk[..., 0], risk[..., 1])
    return risk, assignment


@torch.inference_mode()
def fit_value_tail_path(
    keys: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate by K radius and isolate the least-redundant V token.

    The second center goes to blocks with a large one-center covering radius
    in key space.  Inside such a block, the KV pair whose value is farthest
    from the block value mean becomes a singleton; the other keys share their
    arithmetic mean.  Both decisions use only a sealed block and cost O(SD).
    """
    if keys.shape != values.shape:
        raise ValueError("value-tail placement requires matching K/V shapes")
    block_size = keys.shape[-2]
    if block_size < 2:
        raise ValueError("value-tail placement requires at least two tokens")

    k = keys.float()
    v = values.float()
    k_radius = (k - k.mean(-2, keepdim=True)).norm(dim=-1).amax(-1)
    isolated = (v - v.mean(-2, keepdim=True)).norm(dim=-1).argmax(-1)

    leading = keys.shape[:-2]
    assignment = torch.zeros(
        *leading, 2, block_size, device=keys.device, dtype=torch.uint8
    )
    assignment[..., 1, :].fill_(1)
    assignment[..., 1, :].scatter_(
        -1,
        isolated[..., None],
        torch.zeros_like(isolated[..., None], dtype=torch.uint8),
    )
    return k_radius, assignment


@torch.inference_mode()
def pack_adaptive_centroids(
    keys: torch.Tensor,
    assignment_path: torch.Tensor,
    component_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize only the allocated components, with no padded dot products."""
    n_blocks, block_size, dim = keys.shape[-3:]
    if assignment_path.shape[-3:] != (n_blocks, block_size, block_size):
        raise ValueError("assignment_path shape does not match keys")
    if component_counts.shape != keys.shape[:-2]:
        raise ValueError("component_counts shape does not match keys")
    if torch.any((component_counts < 1) | (component_counts > block_size)):
        raise ValueError("component counts must lie in [1, block_size]")

    chosen = assignment_path.gather(
        -2,
        (component_counts - 1)[..., None, None].expand(
            *component_counts.shape, 1, block_size
        ),
    ).squeeze(-2).long()
    membership = nn.functional.one_hot(
        chosen, num_classes=block_size
    ).float()
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
    isotropic_alpha = trace / (2.0 * dim)
    valid = (
        torch.arange(block_size, device=keys.device)
        < component_counts[..., None]
    )

    leading = keys.shape[:-3]
    groups = math.prod(leading) if leading else 1
    total = int(component_counts.reshape(groups, n_blocks).sum(-1)[0].item())
    totals = component_counts.reshape(groups, n_blocks).sum(-1)
    if torch.any(totals != total):
        raise ValueError("every batch/head group must receive the same budget")
    valid_flat = valid.reshape(groups, n_blocks * block_size)
    packed_centers = centers.reshape(
        groups, n_blocks * block_size, dim
    )[valid_flat].reshape(groups, total, dim)
    packed_populations = populations.reshape(
        groups, n_blocks * block_size
    )[valid_flat].reshape(groups, total)
    packed_alpha = isotropic_alpha.reshape(
        groups, n_blocks * block_size
    )[valid_flat].reshape(groups, total)
    block_ids = torch.arange(n_blocks, device=keys.device)[None, :, None]
    block_ids = block_ids.expand(groups, n_blocks, block_size)
    packed_block_ids = block_ids.reshape(
        groups, n_blocks * block_size
    )[valid_flat].reshape(groups, total)

    return (
        packed_centers.to(keys.dtype).reshape(*leading, total, dim),
        packed_populations.long().reshape(*leading, total),
        packed_alpha.reshape(*leading, total),
        packed_block_ids.reshape(*leading, total),
    )


@torch.inference_mode()
def fit_direct_lse_two_centroids(
    keys: torch.Tensor,
    calibration_queries: torch.Tensor,
    batch_blocks: int = 256,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Globally optimize the calibrated block-LSE loss over 127 partitions.

    ``keys`` is [N, S, D], S <= 8. ``calibration_queries`` is [Q, D] and
    already includes the 1/sqrt(D) attention scale.  The returned quantities
    describe the unique r=1 representation and the globally optimal r=2
    representation: centers, counts, alpha_1, alpha_2, risk_1, risk_2,
    diameter_1, and per-component diameter_2.
    """
    if keys.ndim != 3 or calibration_queries.ndim != 2:
        raise ValueError("direct-LSE optimizer expects [N,S,D] keys and [Q,D] queries")
    block_size, dim = keys.shape[-2:]
    if not 2 <= block_size <= 8:
        raise ValueError("direct-LSE optimizer supports block sizes 2 through 8")
    if calibration_queries.shape[-1] != dim or calibration_queries.shape[0] < 1:
        raise ValueError("invalid direct-LSE calibration query bank")

    device = keys.device
    patterns = torch.arange(
        2 ** (block_size - 1) - 1, device=device, dtype=torch.long
    )
    bit_pos = torch.arange(block_size - 1, device=device)
    tail = ((patterns[:, None] >> bit_pos[None, :]) & 1).bool()
    masks = torch.cat(
        (torch.ones(tail.shape[0], 1, device=device, dtype=torch.bool), tail),
        dim=1,
    )
    mask_f = masks.float()
    count_a = mask_f.sum(-1)
    count_b = float(block_size) - count_a
    counts_all = torch.stack((count_a, count_b), dim=-1)

    q = calibration_queries.float()
    query_norm2 = q.square().sum(-1)
    norm4_sum = query_norm2.square().sum().clamp_min(1e-12)
    outputs: list[list[torch.Tensor]] = [[] for _ in range(8)]

    for start in range(0, keys.shape[0], batch_blocks):
        x = keys[start : start + batch_blocks].float()
        total = x.sum(1)
        sum_a = torch.einsum("cs,bsd->bcd", mask_f, x)
        center_a = sum_a / count_a[None, :, None]
        center_b = (total[:, None] - sum_a) / count_b[None, :, None]
        centers_all = torch.stack((center_a, center_b), dim=2)

        exact_lse = torch.logsumexp(torch.einsum("qd,bsd->qbs", q, x), dim=-1)
        lower_two = torch.logsumexp(
            torch.einsum("qd,bcrd->qbcr", q, centers_all)
            + counts_all.log()[None, None],
            dim=-1,
        )
        gap_two = exact_lse[:, :, None] - lower_two
        alpha_two_all = (
            gap_two * query_norm2[:, None, None]
        ).sum(0) / norm4_sum
        risk_two_all = (
            gap_two
            - query_norm2[:, None, None] * alpha_two_all[None]
        ).square().mean(0)
        best = risk_two_all.argmin(-1)
        rows = torch.arange(x.shape[0], device=device)

        mean = x.mean(1)
        lower_one = torch.einsum("qd,bd->qb", q, mean) + math.log(block_size)
        gap_one = exact_lse - lower_one
        alpha_one = (gap_one * query_norm2[:, None]).sum(0) / norm4_sum
        risk_one = (
            gap_one - query_norm2[:, None] * alpha_one[None]
        ).square().mean(0)

        chosen_mask = masks[best]
        pair_dist = torch.cdist(x, x)
        same_a = chosen_mask[:, :, None] & chosen_mask[:, None, :]
        same_b = (~chosen_mask)[:, :, None] & (~chosen_mask)[:, None, :]
        diameter_a = pair_dist.masked_fill(~same_a, -torch.inf).amax((1, 2))
        diameter_b = pair_dist.masked_fill(~same_b, -torch.inf).amax((1, 2))

        selected = (
            centers_all[rows, best].to(keys.dtype),
            counts_all[best].long(),
            alpha_one,
            alpha_two_all[rows, best],
            risk_one,
            risk_two_all[rows, best],
            pair_dist.amax((1, 2)),
            torch.stack((diameter_a, diameter_b), dim=-1),
        )
        for parts, value in zip(outputs, selected):
            parts.append(value)

    return tuple(torch.cat(parts) for parts in outputs)


@torch.inference_mode()
def fit_block_centroids(
    keys: torch.Tensor,
    n_centroids: int,
    method: str,
    lloyd_steps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition every block and return its exact group means and counts.

    ``keys`` has shape [..., S, D].  The returned tensors have shapes
    [..., R, D] and [..., R].  Counts are strictly positive.
    """
    block_size, dim = keys.shape[-2:]
    if not 1 <= n_centroids <= block_size:
        raise ValueError("n_centroids must lie in [1, block_size]")
    if method not in {"farthest_body", "kmeans", "minimax2"}:
        raise ValueError("method must be farthest_body, kmeans, or minimax2")
    if method in {"farthest_body", "minimax2"} and n_centroids != 2:
        raise ValueError(f"{method} is defined only for two centroids")

    if method == "minimax2":
        centers, counts, _, _ = fit_minimax_two_centroids(keys)
        return centers, counts

    leading = keys.shape[:-2]
    x = keys.reshape(-1, block_size, dim).float()
    n_blocks = x.shape[0]

    if method == "farthest_body":
        mean = x.mean(1, keepdim=True)
        outlier_idx = (x - mean).square().sum(-1).argmax(-1)
        outlier = x[torch.arange(n_blocks, device=x.device), outlier_idx]
        body = (x.sum(1) - outlier) / float(block_size - 1)
        centers = torch.stack((outlier, body), dim=1)
        counts = torch.tensor(
            [1, block_size - 1], device=x.device, dtype=torch.long
        ).view(1, 2).expand(n_blocks, -1)
    else:
        # Deterministic Gonzalez initialisation, followed by a few Lloyd steps.
        mean = x.mean(1, keepdim=True)
        first = (x - mean).square().sum(-1).argmax(-1)
        rows = torch.arange(n_blocks, device=x.device)
        chosen = [x[rows, first]]
        min_dist = (x - chosen[0][:, None]).square().sum(-1)
        for _ in range(1, n_centroids):
            nxt = min_dist.argmax(-1)
            center = x[rows, nxt]
            chosen.append(center)
            min_dist = torch.minimum(
                min_dist, (x - center[:, None]).square().sum(-1)
            )
        centers = torch.stack(chosen, dim=1)

        for _ in range(lloyd_steps):
            assignment = _squared_distances(x, centers).argmin(-1)
            one_hot = nn.functional.one_hot(
                assignment, num_classes=n_centroids
            ).to(x.dtype)
            counts_f = one_hot.sum(1)
            sums = torch.einsum("nsr,nsd->nrd", one_hot, x)
            # Initial centers are distinct data points, so empty clusters are
            # not expected; retaining the old center makes the fallback safe.
            centers = torch.where(
                (counts_f > 0)[..., None],
                sums / counts_f.clamp_min(1)[..., None],
                centers,
            )
        assignment = _squared_distances(x, centers).argmin(-1)
        one_hot = nn.functional.one_hot(
            assignment, num_classes=n_centroids
        ).to(x.dtype)
        counts_f = one_hot.sum(1)
        if (counts_f == 0).any():
            raise RuntimeError("empty centroid after deterministic k-means")
        centers = torch.einsum("nsr,nsd->nrd", one_hot, x) / counts_f[..., None]
        counts = counts_f.long()

    return (
        centers.to(keys.dtype).reshape(*leading, n_centroids, dim),
        counts.reshape(*leading, n_centroids),
    )


class CentroidLSERouterShadowKVCache(ShadowKVCache):
    """Replace one block mean by an equal-form mixture of R centroids."""

    def __init__(
        self,
        *args,
        n_centroids: int = 2,
        centroid_method: str = "kmeans",
        split_fraction: float = 1.0,
        calibration_tokens: int = 8,
        self_lse_temperatures: tuple[float, ...] = (1.0, 1.5, 2.0),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 1 <= n_centroids <= self.chunk_size:
            raise ValueError("router centroids must fit inside a block")
        self.n_centroids = n_centroids
        self.centroid_method = centroid_method
        if centroid_method == "direct_lse2" and n_centroids != 2:
            raise ValueError("direct_lse2 requires exactly two centroids")
        if (
            centroid_method == "self_lse_adaptive_iso"
            and n_centroids != self.chunk_size
        ):
            raise ValueError(
                "self_lse_adaptive_iso requires n_centroids == chunk_size"
            )
        max_split_fraction = (
            float(self.chunk_size - 1)
            if centroid_method == "self_lse_adaptive_iso"
            else 1.0
        )
        if not 0.0 <= split_fraction <= max_split_fraction:
            raise ValueError(
                "split_fraction must lie in "
                f"[0,{max_split_fraction:g}] for {centroid_method}"
            )
        binary_adaptive_methods = {
            "minimax2",
            "minimax2_sse",
            "scatter2",
            "cosine2",
            "self_lse2",
            "self_lse2_peak",
            "self_lse2_iso",
            "direct_lse2",
        }
        if split_fraction < 1.0 and centroid_method not in (
            binary_adaptive_methods | {"self_lse_adaptive_iso"}
        ):
            raise ValueError(
                "split_fraction is unsupported for this centroid method"
            )
        if split_fraction < 1.0 and (
            centroid_method in binary_adaptive_methods and n_centroids != 2
        ):
            raise ValueError("binary adaptive splitting requires two centroids")
        self.split_fraction = split_fraction
        if calibration_tokens < 1:
            raise ValueError("calibration_tokens must be positive")
        self.calibration_tokens = calibration_tokens
        if not self_lse_temperatures or any(
            value <= 0 for value in self_lse_temperatures
        ):
            raise ValueError("self_lse_temperatures must be positive")
        self.self_lse_temperatures = tuple(self_lse_temperatures)
        self.router_centroids: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.router_log_counts: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.router_alpha: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.router_diameter: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.router_isotropic_alpha: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.router_allocation_gain: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.router_split_mask: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.router_component_block: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.router_component_count: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]
        self.router_risk_path: list[torch.Tensor | None] = [
            None for _ in range(self.num_layers)
        ]

    def print_stats(self):
        super().print_stats()
        print(
            "ShadowKV_CENTROID_LSE | "
            f"centroids {self.n_centroids} | method {self.centroid_method} | "
            f"split_fraction {self.split_fraction} | "
            f"calibration_tokens {self.calibration_tokens} | "
            f"self_lse_temperatures {self.self_lse_temperatures}"
        )

    def prefill_kv_cache(
        self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query: torch.Tensor = None,
    ):
        super().prefill_kv_cache(
            new_v_cache, layer_idx, key_states_roped, query
        )
        eligible = self.chunks * self.chunk_size
        blocks = key_states_roped[:, :, :eligible].view(
            self.batch_size,
            self.num_key_value_heads,
            self.chunks,
            self.chunk_size,
            self.head_dim,
        )
        rest_idx = self.k_landmark_idx[layer_idx]
        compact_blocks = blocks.gather(
            dim=2,
            index=rest_idx[..., None, None].expand(
                -1, -1, -1, self.chunk_size, self.head_dim
            ),
        )
        allocation_gain = None
        alpha = None
        diameter = None
        isotropic_alpha = None
        component_block = None
        component_count = None
        risk_path = None
        if self.centroid_method == "self_lse_adaptive_iso":
            if self.split_fraction <= 1.0:
                (
                    risk_path,
                    assignment_path,
                    component_count,
                ) = fit_lazy_exact_self_lse_allocation(
                    compact_blocks,
                    extra_fraction=self.split_fraction,
                    temperatures=self.self_lse_temperatures,
                )
                # Exact prefix blocks do not compete for retrieval and must
                # not consume the adaptive component budget.  The historical
                # prefix4 variant added the exact prefix after fitting, which
                # silently spent part of the 25% quota on blocks that were
                # subsequently masked from top-k.
                prefix_chunks = int(getattr(self, "prefix_chunks", 0))
                if prefix_chunks:
                    candidate_count = compact_blocks.shape[2] - prefix_chunks
                    if candidate_count <= 0:
                        raise ValueError("exact prefix leaves no router candidates")
                    n_extra = int(round(self.split_fraction * candidate_count))
                    component_count = torch.ones_like(component_count)
                    if n_extra:
                        candidate_risk = risk_path[..., prefix_chunks:, 0]
                        chosen = candidate_risk.topk(n_extra, dim=-1).indices
                        component_count[..., prefix_chunks:].scatter_(
                            -1, chosen, 2
                        )
            else:
                risk_path, assignment_path = (
                    fit_agglomerative_self_lse_paths(
                        compact_blocks,
                        temperatures=self.self_lse_temperatures,
                    )
                )
                component_count = allocate_minimax_centroid_counts(
                    risk_path, self.split_fraction
                )
            (
                centers,
                counts,
                isotropic_alpha,
                component_block,
            ) = pack_adaptive_centroids(
                compact_blocks, assignment_path, component_count
            )
            allocation_gain = risk_path[..., 0] - risk_path[..., 1]
            split = component_count > 1
        elif self.centroid_method == "direct_lse2":
            if query is None or query.shape[-2] < self.calibration_tokens + 1:
                raise ValueError(
                    "direct_lse2 requires the prompt query bank, including one held-out final query"
                )
            prompt_q = query[:, :, -(self.calibration_tokens + 1) : -1].float()
            prompt_q = prompt_q.view(
                self.batch_size,
                self.num_key_value_heads,
                self.num_key_value_groups,
                self.calibration_tokens,
                self.head_dim,
            ) / math.sqrt(self.head_dim)
            result_by_batch_head = []
            for batch in range(self.batch_size):
                result_by_head = []
                for head in range(self.num_key_value_heads):
                    result_by_head.append(
                        fit_direct_lse_two_centroids(
                            compact_blocks[batch, head],
                            prompt_q[batch, head].reshape(-1, self.head_dim),
                        )
                    )
                result_by_batch_head.append(result_by_head)

            def stack_result(index):
                return torch.stack(
                    [
                        torch.stack(
                            [result_by_batch_head[b][h][index] for h in range(self.num_key_value_heads)]
                        )
                        for b in range(self.batch_size)
                    ]
                )

            centers = stack_result(0)
            counts = stack_result(1)
            alpha_one = stack_result(2)
            alpha_two = stack_result(3)
            risk_one = stack_result(4)
            risk_two = stack_result(5)
            diameter_one = stack_result(6)
            diameter_two = stack_result(7)
            allocation_gain = risk_one - risk_two
        elif self.centroid_method in {"minimax2", "minimax2_sse"}:
            centers, counts, radius_one, radius_two = fit_minimax_two_centroids(
                compact_blocks
            )
            if self.centroid_method == "minimax2":
                allocation_gain = radius_one - radius_two
            else:
                mean = compact_blocks.float().mean(dim=-2)
                one_sse = (
                    compact_blocks.float() - mean[..., None, :]
                ).square().sum(-1).mean(-1)
                two_sse = (
                    compact_blocks.float()[..., :, None, :]
                    - centers.float()[..., None, :, :]
                ).square().sum(-1).amin(-1).mean(-1)
                allocation_gain = (one_sse - two_sse).clamp_min(0)
        elif self.centroid_method in {"scatter2", "cosine2"}:
            centers, counts, _, _, allocation_gain = fit_key_only_two_centroids(
                compact_blocks,
                objective={"scatter2": "scatter", "cosine2": "cosine"}[
                    self.centroid_method
                ],
            )
        elif self.centroid_method in {
            "self_lse2",
            "self_lse2_peak",
            "self_lse2_iso",
        }:
            fitted = fit_self_lse_two_centroids(
                compact_blocks,
                temperatures=self.self_lse_temperatures,
                return_isotropic_alpha=self.centroid_method == "self_lse2_iso",
            )
            if self.centroid_method == "self_lse2_iso":
                centers, counts, _, _, allocation_gain, isotropic_two = fitted
            else:
                centers, counts, _, _, allocation_gain = fitted
            if self.centroid_method == "self_lse2_peak":
                unit = nn.functional.normalize(
                    compact_blocks.float(), dim=-1, eps=1e-12
                )
                self_logits = torch.einsum(
                    "...jd,...id->...ji", unit, compact_blocks.float()
                )
                allocation_gain = torch.softmax(self_logits, dim=-1).amax(
                    dim=(-2, -1)
                )
        else:
            centers, counts = fit_block_centroids(
                compact_blocks,
                n_centroids=self.n_centroids,
                method=self.centroid_method,
            )

        if self.centroid_method == "self_lse_adaptive_iso":
            pass
        elif self.split_fraction < 1.0:
            n_blocks = compact_blocks.shape[2]
            n_split = int(round(self.split_fraction * n_blocks))
            split = torch.zeros_like(allocation_gain, dtype=torch.bool)
            if n_split:
                chosen = allocation_gain.topk(n_split, dim=-1).indices
                split.scatter_(dim=-1, index=chosen, value=True)
            if n_split < n_blocks:
                mean = compact_blocks.mean(dim=-2)
                centers = torch.where(
                    split[..., None, None],
                    centers,
                    torch.stack((mean, torch.zeros_like(mean)), dim=-2),
                )
                unsplit_counts = torch.zeros_like(counts)
                unsplit_counts[..., 0] = self.chunk_size
                counts = torch.where(split[..., None], counts, unsplit_counts)
                if self.centroid_method == "direct_lse2":
                    alpha = torch.where(split, alpha_two, alpha_one)
                    diameter = torch.where(
                        split[..., None],
                        diameter_two,
                        torch.stack(
                            (diameter_one, torch.zeros_like(diameter_one)), dim=-1
                        ),
                    )
            if self.centroid_method == "self_lse2_iso":
                mean_for_alpha = compact_blocks.float().mean(dim=-2)
                trace_one = (
                    compact_blocks.float() - mean_for_alpha[..., None, :]
                ).square().sum(-1).mean(-1)
                isotropic_one = trace_one / (2.0 * self.head_dim)
                isotropic_alpha = torch.where(
                    split[..., None],
                    isotropic_two,
                    torch.stack(
                        (isotropic_one, torch.zeros_like(isotropic_one)), dim=-1
                    ),
                )
        elif self.centroid_method == "direct_lse2":
            alpha = alpha_two
            diameter = diameter_two
            split = counts[..., 1] > 0
        elif self.centroid_method == "self_lse2_iso":
            isotropic_alpha = isotropic_two
            split = counts[..., 1] > 0
        else:
            split = counts[..., 1] > 0
        self.router_centroids[layer_idx] = centers
        self.router_log_counts[layer_idx] = counts.float().log().to(centers.dtype)
        self.router_alpha[layer_idx] = alpha
        self.router_diameter[layer_idx] = diameter
        self.router_isotropic_alpha[layer_idx] = isotropic_alpha
        self.router_allocation_gain[layer_idx] = allocation_gain
        self.router_split_mask[layer_idx] = split
        self.router_component_block[layer_idx] = component_block
        # Counts are diagnostic metadata only.  Keep one byte per block rather
        # than an int64 tensor; the decode path uses the packed component map.
        self.router_component_count[layer_idx] = (
            component_count.to(torch.uint8)
            if component_count is not None
            else None
        )
        # The full N x S risk path is needed only while solving the allocation.
        # Do not retain it for every layer during generation.
        self.router_risk_path[layer_idx] = None

    def get_retrieval_position_ids(self, layer_idx, query_states):
        self.incoming_q_len = query_states.shape[-2]
        query = query_states.view(
            -1,
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.incoming_q_len,
            self.head_dim,
        )
        if self.centroid_method == "self_lse_adaptive_iso":
            component_logits = torch.einsum(
                "bhgqd,bhmd->bhgqm",
                query,
                self.router_centroids[layer_idx],
            ).float() / math.sqrt(self.head_dim)
            component_logits = component_logits + self.router_log_counts[
                layer_idx
            ].float()[:, :, None, None, :]
            query_norm2 = query.float().square().sum(-1) / self.head_dim
            component_logits = component_logits + query_norm2[..., None] * (
                self.router_isotropic_alpha[layer_idx][
                    :, :, None, None, :
                ].float()
            )
            component_block = self.router_component_block[layer_idx]
            index = component_block[:, :, None, None, :].expand_as(
                component_logits
            )
            n_blocks = self.k_landmark_idx[layer_idx].shape[-1]
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
            centered = component_logits - block_max.gather(-1, index)
            block_sum = torch.zeros_like(block_max)
            block_sum.scatter_add_(dim=-1, index=index, src=centered.exp())
            block_logits = block_max + block_sum.log()
        else:
            component_logits = torch.einsum(
                "bhgqd,bhcrd->bhgqcr",
                query,
                self.router_centroids[layer_idx],
            ).float() / math.sqrt(self.head_dim)
            component_logits = component_logits + self.router_log_counts[
                layer_idx
            ].float()[:, :, None, None, :, :]
            if self.centroid_method == "self_lse2_iso":
                query_norm2 = query.float().square().sum(-1) / self.head_dim
                component_logits = component_logits + query_norm2[
                    ..., None, None
                ] * (
                    self.router_isotropic_alpha[layer_idx][
                        :, :, None, None, :, :
                    ].float()
                )
            block_logits = torch.logsumexp(component_logits, dim=-1)
        if self.centroid_method == "direct_lse2":
            query_norm2 = query.float().square().sum(-1) / self.head_dim
            corrected = block_logits + query_norm2[..., None] * self.router_alpha[
                layer_idx
            ][:, :, None, None, :]
            upper = torch.logsumexp(
                component_logits
                + query_norm2[..., None, None]
                * self.router_diameter[layer_idx][
                    :, :, None, None, :, :
                ].float().square()
                / 8.0,
                dim=-1,
            )
            block_logits = torch.minimum(corrected, upper)
        block_logits = self._mask_router_block_logits(
            layer_idx, block_logits
        ).to(self.dtype)
        chunk_attn = nn.functional.softmax(
            block_logits, dim=-1, dtype=torch.float32
        ).to(self.dtype)
        chunk_attn = chunk_attn.sum(dim=-2)
        if self.num_key_value_groups > 1:
            chunk_attn = torch.max(chunk_attn, dim=-2).values
        selected_chunks = self._select_router_chunks(layer_idx, chunk_attn)
        self.selected_chunk_idx[layer_idx].copy_(
            selected_chunks, non_blocking=True
        )
        return (
            selected_chunks.unsqueeze(-1) * self.chunk_size
            + torch.arange(self.chunk_size, device=chunk_attn.device).view(
                1, 1, 1, -1
            )
        ).view(self.batch_size, self.num_key_value_heads, -1)

    def _mask_router_block_logits(self, layer_idx, block_logits):
        """Mask exact-only blocks before per-query-head normalization."""
        del layer_idx
        return block_logits

    def _select_router_chunks(self, layer_idx, chunk_attn):
        """Map the highest-scoring compact router entries to source blocks.

        Kept as a hook so variants with an exact side cache can exclude those
        blocks from dynamic retrieval without duplicating the scoring path.
        """
        merged_results = torch.topk(
            chunk_attn, k=self.select_sets, dim=-1
        ).indices
        return self.k_landmark_idx[layer_idx].gather(
            dim=-1, index=merged_results
        )

    def clear(self):
        super().clear()
        self.router_centroids = [None for _ in range(self.num_layers)]
        self.router_log_counts = [None for _ in range(self.num_layers)]
        self.router_alpha = [None for _ in range(self.num_layers)]
        self.router_diameter = [None for _ in range(self.num_layers)]
        self.router_isotropic_alpha = [None for _ in range(self.num_layers)]
        self.router_allocation_gain = [None for _ in range(self.num_layers)]
        self.router_split_mask = [None for _ in range(self.num_layers)]
        self.router_component_block = [None for _ in range(self.num_layers)]
        self.router_component_count = [None for _ in range(self.num_layers)]
        self.router_risk_path = [None for _ in range(self.num_layers)]
