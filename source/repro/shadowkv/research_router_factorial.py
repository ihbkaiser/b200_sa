#!/usr/bin/env python3
"""Factor the remaining adaptive-centroid routing error into four stages.

This is an offline diagnostic, not a production selector.  Future decode
queries are used only to construct counterfactual oracle allocators.  Every
deployable row remains query-free at prefill and uses only the current decode
query while ranking blocks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))

from models.adaptive_centroid_streaming_cache import (  # noqa: E402
    _padded_adaptive_centroids,
)
from models.centroid_router_cache import (  # noqa: E402
    allocate_concave_marginal_counts,
    evaluate_self_lse_cvar_path,
    fit_agglomerative_self_lse_paths,
    fit_residual_tail_lse_paths,
)


def top_mask(score: torch.Tensor, count: int) -> torch.Tensor:
    return torch.zeros_like(score, dtype=torch.bool).scatter_(
        0, score.topk(min(count, score.shape[0]), dim=0).indices, True
    )


def metrics(score: torch.Tensor, exact: torch.Tensor, count: int) -> dict[str, float]:
    pred = top_mask(score, count)
    truth = top_mask(exact, count)
    weight = torch.exp(exact - exact.max(0, keepdim=True).values)
    centered_score = score - score.mean(0, keepdim=True)
    centered_exact = exact - exact.mean(0, keepdim=True)
    correlation = (centered_score * centered_exact).sum(0) / (
        centered_score.square().sum(0).sqrt()
        * centered_exact.square().sum(0).sqrt()
    ).clamp_min(1e-12)
    return {
        "overlap": float((pred & truth).sum(0).float().div(count).mean()),
        "mass_recall": float(((weight * pred).sum(0) / weight.sum(0)).mean()),
        "best_hit": float(
            pred.gather(0, exact.argmax(0, keepdim=True)).float().mean()
        ),
        "pearson": float(correlation.mean()),
        "mae_centered": float(
            (centered_score - centered_exact).abs().mean()
        ),
    }


def materialize_curve(
    keys: torch.Tensor,
    queries: torch.Tensor,
    path: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and isotropic-second-order LSE for every order 1..S."""
    blocks, size, dim = keys.shape
    steps = queries.shape[0]
    raw = torch.empty(blocks, steps, size, device=keys.device)
    isotropic = torch.empty_like(raw)
    shaped_keys = keys[None, None]
    shaped_path = path[None, None]
    for order in range(1, size + 1):
        counts = torch.full(
            (1, 1, blocks), order, device=keys.device, dtype=torch.long
        )
        centers, log_counts, alpha = _padded_adaptive_centroids(
            shaped_keys, shaped_path, counts, size
        )
        centers = centers[0, 0]
        log_counts = log_counts[0, 0]
        alpha = alpha[0, 0]
        logits = torch.einsum("td,prd->ptr", queries, centers) / math.sqrt(dim)
        logits = logits + log_counts[:, None]
        valid = torch.arange(size, device=keys.device)[None] < order
        logits = logits.masked_fill(~valid[:, None], float("-inf"))
        raw[..., order - 1] = torch.logsumexp(logits, dim=-1)
        qnorm2 = queries.square().sum(-1) / dim
        isotropic[..., order - 1] = torch.logsumexp(
            logits + qnorm2[None, :, None] * alpha[:, None], dim=-1
        )
    return raw, isotropic


def gather_order(curve: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return curve.gather(
        -1, (counts - 1)[:, None, None].expand(-1, curve.shape[1], 1)
    ).squeeze(-1)


def fit_ward_scatter_path(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nested Ward path: greedily minimize the increase in within-cluster SSE."""
    blocks, size, _ = keys.shape
    sums = keys.float().clone()
    counts = torch.ones(blocks, size, device=keys.device)
    labels = torch.arange(size, device=keys.device).expand(blocks, -1).clone()
    path = torch.empty(
        blocks, size, size, device=keys.device, dtype=torch.uint8
    )
    path[:, size - 1] = labels.to(torch.uint8)
    risk = torch.zeros(blocks, size, device=keys.device)
    total = torch.zeros(blocks, device=keys.device)
    energy = keys.float().square().sum((1, 2)).clamp_min(1e-12)
    rows = torch.arange(blocks, device=keys.device)
    for order in range(size, 1, -1):
        left, right = torch.triu_indices(order, order, 1, device=keys.device)
        mean_left = sums[:, left] / counts[:, left, None]
        mean_right = sums[:, right] / counts[:, right, None]
        factor = (
            counts[:, left] * counts[:, right]
            / (counts[:, left] + counts[:, right])
        )
        increase = factor * (mean_left - mean_right).square().sum(-1)
        best = increase.argmin(-1)
        chosen_left, chosen_right = left[best], right[best]
        total = total + increase[rows, best]
        risk[:, order - 2] = total / energy
        merged_sum = sums[rows, chosen_left] + sums[rows, chosen_right]
        merged_count = counts[rows, chosen_left] + counts[rows, chosen_right]
        output = torch.arange(order - 1, device=keys.device)[None]
        old = output + (output >= chosen_right[:, None])
        sums = sums.gather(-2, old[..., None].expand(-1, -1, sums.shape[-1]))
        counts = counts.gather(-1, old)
        sums[rows, chosen_left] = merged_sum
        counts[rows, chosen_left] = merged_count
        labels = torch.where(
            labels == chosen_right[:, None], chosen_left[:, None], labels
        )
        labels = torch.where(
            labels > chosen_right[:, None], labels - 1, labels
        )
        path[:, order - 2] = labels.to(torch.uint8)
    return risk, path


def path_family(
    keys: torch.Tensor,
    anchor: torch.Tensor | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    shaped = keys[None, None]
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, mode, beta in (
        ("robust_trim25", "trimmed_gap", 0.25),
        ("minimax_gap", "max_gap", 4.0),
        ("mean_relative", "mean_relative", 4.0),
        ("soft_relative", "relative_lme", 4.0),
        ("weighted_relative", "weighted_relative_lme", 4.0),
    ):
        placement_risk, path = fit_agglomerative_self_lse_paths(
            shaped,
            temperatures=(1.0,),
            cost_mode=mode,
            cost_beta=beta,
        )
        result[name] = (placement_risk[0, 0], path[0, 0])
    for name, selection in (
        ("isolate_residual", "residual"),
        ("isolate_angular", "angular"),
        ("isolate_norm", "norm"),
    ):
        placement_risk, path = fit_residual_tail_lse_paths(
            shaped,
            temperatures=(1.0,),
            cost_mode="trimmed_gap",
            cost_beta=0.25,
            selection_mode=selection,
        )
        result[name] = (placement_risk[0, 0], path[0, 0])
    result["ward_scatter"] = fit_ward_scatter_path(keys)
    if anchor is not None:
        placement_risk, path = fit_agglomerative_self_lse_paths(
            shaped,
            temperatures=(1.0,),
            cost_mode="max_gap",
            cost_beta=4.0,
            extra_proxy_queries=anchor[None, None],
        )
        result["minimax_qanchor"] = (
            placement_risk[0, 0], path[0, 0]
        )
    return result


def allocation_family(
    placement_risk: torch.Tensor,
    self_tail: torch.Tensor,
    self_mean: torch.Tensor,
    raw_curve: torch.Tensor,
    exact: torch.Tensor,
    extra: float,
) -> dict[str, torch.Tensor]:
    gap = (exact[..., None] - raw_curve).clamp_min(0)
    exact_probability = torch.softmax(exact, dim=0)
    missed_mass = (
        exact_probability[..., None]
        * (1.0 - torch.exp(-gap)).clamp_min(0)
    ).mean(1)
    anchor_gap = gap[:, 0]
    anchor_probability = exact_probability[:, 0]
    anchor_missed_mass = anchor_probability[:, None] * (
        1.0 - torch.exp(-anchor_gap)
    ).clamp_min(0)
    return {
        "tail_cvar25": allocate_concave_marginal_counts(self_tail, extra),
        "self75_anchor25": allocate_concave_marginal_counts(
            0.75 * self_tail + 0.25 * anchor_gap, extra
        ),
        "self50_anchor50": allocate_concave_marginal_counts(
            0.5 * self_tail + 0.5 * anchor_gap, extra
        ),
        "self25_anchor75": allocate_concave_marginal_counts(
            0.25 * self_tail + 0.75 * anchor_gap, extra
        ),
        "anchor_missed_mass": allocate_concave_marginal_counts(
            anchor_missed_mass, extra
        ),
        "self_mean": allocate_concave_marginal_counts(self_mean, extra),
        "placement_risk": allocate_concave_marginal_counts(
            placement_risk, extra
        ),
        # Rate controls used to price mixed-precision metadata.  Two INT8
        # centers cost less center storage than the canonical 1.25 FP16
        # centers; these rows are not compared as equal center counts.
        "uniform_r2": torch.full_like(placement_risk[..., 0], 2).long(),
        "uniform_r3": torch.full_like(placement_risk[..., 0], 3).long(),
        # Counterfactual ceilings: these consume future decode queries and are
        # never deployable.  They quantify whether allocation is worth more work.
        "oracle_gap": allocate_concave_marginal_counts(gap.mean(1), extra),
        "oracle_mass": allocate_concave_marginal_counts(missed_mass, extra),
    }


def add_row(
    rows: list[dict], common: dict, stage: str, placement: str,
    allocation: str, score_name: str, score: torch.Tensor,
    exact: torch.Tensor, budget: int, counts: torch.Tensor,
    oracle_counts: torch.Tensor,
) -> None:
    count = budget // common["block_size"]
    rows.append({
        **common,
        "stage": stage,
        "placement": placement,
        "allocation": allocation,
        "score": score_name,
        "budget": budget,
        "mean_r": float(counts.float().mean()),
        "count_exact_match": float((counts == oracle_counts).float().mean()),
        "count_mae_vs_oracle": float(
            (counts.float() - oracle_counts.float()).abs().mean()
        ),
        **metrics(score, exact, count),
    })


@torch.inference_mode()
def evaluate_pair(
    blocks: torch.Tensor,
    query_bank: torch.Tensor,
    extra: float,
    budgets: list[int],
    device: str,
    common: dict,
) -> list[dict]:
    keys = blocks.to(device=device, dtype=torch.float32)
    query = query_bank.to(device=device, dtype=torch.float32).mean(0)
    dim = keys.shape[-1]
    exact = torch.logsumexp(
        torch.einsum("td,psd->pts", query, keys) / math.sqrt(dim), dim=-1
    )
    rows: list[dict] = []
    # The first decode query is the final-prefix query that produced the first
    # generated token.  It is causal at index construction; later decode
    # queries remain held out and test whether the anchor generalizes.
    anchor = (query[:1] / math.sqrt(dim)).expand(keys.shape[0], -1, -1)
    families = path_family(keys, anchor)

    for placement, (placement_risk, path) in families.items():
        extra_proxy = anchor if placement == "minimax_qanchor" else None
        shaped = keys[None, None]
        shaped_path = path[None, None]
        self_tail = evaluate_self_lse_cvar_path(
            shaped, shaped_path, temperatures=(1.0,), tail_fraction=0.25,
            extra_proxy_queries=(
                extra_proxy[None, None] if extra_proxy is not None else None
            ),
        )[0, 0]
        self_mean = evaluate_self_lse_cvar_path(
            shaped, shaped_path, temperatures=(1.0,), tail_fraction=1.0,
            extra_proxy_queries=(
                extra_proxy[None, None] if extra_proxy is not None else None
            ),
        )[0, 0]
        raw_curve, alpha_curve = materialize_curve(keys, query, path)
        allocations = allocation_family(
            placement_risk, self_tail, self_mean, raw_curve, exact, extra
        )
        oracle_counts = allocations["oracle_mass"]

        # Placement and allocation factorial: use one fixed scoring rule so
        # differences cannot be attributed to a second changing layer.
        for allocation, counts in allocations.items():
            raw = gather_order(raw_curve, counts)
            selected_risk = self_tail.gather(
                1, (counts - 1)[:, None]
            ).squeeze(1)
            score = raw + 0.25 * selected_risk[:, None]
            stage = "placement" if allocation == "tail_cvar25" else "allocation"
            for budget in budgets:
                add_row(
                    rows, common, stage, placement, allocation,
                    "risk_beta0.25", score, exact, budget, counts,
                    oracle_counts,
                )

        if placement not in {"robust_trim25", "minimax_gap"}:
            continue
        counts = allocations["tail_cvar25"]
        raw = gather_order(raw_curve, counts)
        alpha = gather_order(alpha_curve, counts)
        selected_tail = self_tail.gather(
            1, (counts - 1)[:, None]
        ).squeeze(1)
        selected_mean = self_mean.gather(
            1, (counts - 1)[:, None]
        ).squeeze(1)
        selected_place = placement_risk.gather(
            1, (counts - 1)[:, None]
        ).squeeze(1)
        temperature2 = query.square().sum(-1) / dim
        inverse_order = counts.float().reciprocal()[:, None]
        score_family = {
            "raw": raw,
            "isotropic": alpha,
            "tail_beta0.125": raw + 0.125 * selected_tail[:, None],
            "tail_beta0.25": raw + 0.25 * selected_tail[:, None],
            "tail_beta0.375": raw + 0.375 * selected_tail[:, None],
            "tail_beta0.5": raw + 0.5 * selected_tail[:, None],
            "tail_beta0.75": raw + 0.75 * selected_tail[:, None],
            "tail_beta1": raw + selected_tail[:, None],
            "mean_beta0.25": raw + 0.25 * selected_mean[:, None],
            "placement_beta0.25": raw + 0.25 * selected_place[:, None],
            # A count-only missing-mass prior directly counteracts the
            # measured variable-resolution bias.  It costs no metadata: r is
            # already stored for the packed component index.
            "count_inv0.125": raw + 0.125 * inverse_order,
            "count_inv0.25": raw + 0.25 * inverse_order,
            "count_inv0.5": raw + 0.5 * inverse_order,
            "tail0.25_count_inv0.125": (
                raw + 0.25 * selected_tail[:, None] + 0.125 * inverse_order
            ),
            "tail0.25_count_inv0.25": (
                raw + 0.25 * selected_tail[:, None] + 0.25 * inverse_order
            ),
            "tail0.25_count_inv0.5": (
                raw + 0.25 * selected_tail[:, None] + 0.5 * inverse_order
            ),
            # Self-key risks are measured with unit-norm proxy queries.  The
            # local log-MGF expansion predicts quadratic scaling with the
            # actual normalized-query temperature.
            "tail_qnorm2_beta0.125": (
                raw + 0.125 * selected_tail[:, None] * temperature2[None]
            ),
            "tail_qnorm2_beta0.25": (
                raw + 0.25 * selected_tail[:, None] * temperature2[None]
            ),
            "tail_qnorm2_beta0.5": (
                raw + 0.5 * selected_tail[:, None] * temperature2[None]
            ),
        }
        for score_name, score in score_family.items():
            for budget in budgets:
                add_row(
                    rows, common, "score", placement, "tail_cvar25",
                    score_name, score, exact, budget, counts, oracle_counts,
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--extra", type=float, default=0.25)
    ap.add_argument("--budgets", default="512,2048")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()
    budgets = [int(item) for item in args.budgets.split(",")]
    rows: list[dict] = []
    for filename in args.inputs:
        payload = torch.load(filename, map_location="cpu", weights_only=False)
        pairs = sorted(set(payload["blocks"]) & set(payload["queries"]))
        if args.max_pairs:
            pairs = pairs[: args.max_pairs]
        for pair in pairs:
            print(Path(filename).name, pair, flush=True)
            common = {
                "input": filename,
                "sample": payload.get("sample_index"),
                "pair": str(pair),
                "block_size": int(payload.get("block_size", 8)),
            }
            rows.extend(evaluate_pair(
                payload["blocks"][pair], payload["queries"][pair],
                args.extra, budgets, args.device, common,
            ))

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    group_fields = ("stage", "placement", "allocation", "score", "budget")
    metric_fields = (
        "mean_r", "count_exact_match", "count_mae_vs_oracle", "overlap",
        "mass_recall", "best_hit", "pearson", "mae_centered",
    )
    summary: list[dict] = []
    groups = sorted({tuple(row[field] for field in group_fields) for row in rows})
    for group in groups:
        selected = [
            row for row in rows
            if tuple(row[field] for field in group_fields) == group
        ]
        summary.append({
            **dict(zip(group_fields, group)),
            "pairs": len(selected),
            **{
                field: sum(float(row[field]) for row in selected) / len(selected)
                for field in metric_fields
            },
        })
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"saved {args.output}: {len(rows)} rows, {len(summary)} summaries")


if __name__ == "__main__":
    main()
