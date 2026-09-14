#!/usr/bin/env python3
"""Measure whether variable-resolution block summaries bias decode routing.

The probe reconstructs the current robust-trimmed placement + tail-CVaR
allocation from raw post-RoPE block keys, then scores the same decode queries
with one center, the allocated number of centers, and exact token LSE.
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

from models.centroid_router_cache import (  # noqa: E402
    allocate_concave_marginal_counts,
    evaluate_self_lse_cvar_path,
    fit_agglomerative_self_lse_paths,
)


def aggregate(x: torch.Tensor, mode: str) -> torch.Tensor:
    """[P,H,T] -> [P,T]."""
    if mode == "max":
        return x.max(1).values
    if mode == "mean":
        return torch.logsumexp(x, 1) - math.log(x.shape[1])
    raise ValueError(mode)


def top_mask(score: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros_like(score, dtype=torch.bool).scatter_(
        0, score.topk(min(n, score.shape[0]), dim=0).indices, True
    )


def top_metrics(score: torch.Tensor, exact: torch.Tensor, n: int) -> dict[str, float]:
    pred = top_mask(score, n)
    truth = top_mask(exact, n)
    shifted = (exact - exact.max(0, keepdim=True).values).exp()
    return {
        "overlap": float((pred & truth).sum(0).float().div(n).mean()),
        "mass_recall": float(((shifted * pred).sum(0) / shifted.sum(0)).mean()),
        "best_hit": float(pred.gather(0, exact.argmax(0, keepdim=True)).float().mean()),
    }


@torch.inference_mode()
def evaluate(
    blocks: torch.Tensor, queries: torch.Tensor, extra: float, device: str,
    trim_fraction: float, tail_fraction: float,
):
    k = blocks.to(device=device, dtype=torch.float32)
    q = queries.to(device=device, dtype=torch.float32)
    p, c, d = k.shape
    shaped = k[None, None]
    trimmed_gap, path = fit_agglomerative_self_lse_paths(
        shaped, temperatures=(1.0,), cost_mode="trimmed_gap", cost_beta=trim_fraction
    )
    risk = evaluate_self_lse_cvar_path(
        shaped, path, temperatures=(1.0,), tail_fraction=tail_fraction
    )[0, 0]
    mean_gap = evaluate_self_lse_cvar_path(
        shaped, path, temperatures=(1.0,), tail_fraction=1.0
    )[0, 0]
    counts = allocate_concave_marginal_counts(risk[None, None], extra)[0, 0]
    labels = path[0, 0].gather(
        1, (counts - 1)[:, None, None].expand(p, 1, c)
    ).squeeze(1).long()
    membership = F.one_hot(labels, num_classes=c).float()
    population = membership.sum(1)
    centers = torch.einsum("psc,psd->pcd", membership, k) / population.clamp_min(1)[..., None]
    valid = torch.arange(c, device=k.device)[None] < counts[:, None]
    component = torch.einsum("htd,pcd->phtc", q, centers) / math.sqrt(d)
    component = component + population.clamp_min(1).log()[:, None, None]
    component = component.masked_fill(~valid[:, None, None], float("-inf"))
    variable = torch.logsumexp(component, -1)
    one = math.log(c) + torch.einsum("htd,pd->pht", q, k.mean(1)) / math.sqrt(d)
    exact = torch.logsumexp(
        torch.einsum("htd,psd->phts", q, k) / math.sqrt(d), -1
    )
    # Query-mean-first is not recoverable by reducing per-query logits after
    # the fact: LSE is nonlinear.  Recompute all three scores from the single
    # mean query used by the deployed q-mean router.
    qmean = q.mean(0, keepdim=True)
    qmean_component = torch.einsum("htd,pcd->phtc", qmean, centers) / math.sqrt(d)
    qmean_component = qmean_component + population.clamp_min(1).log()[:, None, None]
    qmean_component = qmean_component.masked_fill(
        ~valid[:, None, None], float("-inf")
    )
    qmean_variable = torch.logsumexp(qmean_component, -1).squeeze(1)
    qmean_one = (
        math.log(c)
        + torch.einsum("htd,pd->pht", qmean, k.mean(1)) / math.sqrt(d)
    ).squeeze(1)
    qmean_exact = torch.logsumexp(
        torch.einsum("htd,psd->phts", qmean, k) / math.sqrt(d), -1
    ).squeeze(1)
    chosen_risk = risk.gather(1, (counts - 1)[:, None]).squeeze(1)
    chosen_mean_gap = mean_gap.gather(1, (counts - 1)[:, None]).squeeze(1)
    chosen_trimmed_gap = trimmed_gap[0, 0].gather(
        1, (counts - 1)[:, None]
    ).squeeze(1)
    return (
        counts, chosen_risk, chosen_mean_gap, chosen_trimmed_gap,
        one, variable, exact, (qmean_one, qmean_variable, qmean_exact),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--extra", type=float, default=0.25)
    ap.add_argument("--trim", type=float, default=0.25)
    ap.add_argument("--tail", type=float, default=0.25)
    ap.add_argument("--corrections", default="0.125,0.25,0.375,0.5,0.75,1")
    ap.add_argument("--budgets", default="512,2048")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]
    corrections = [float(x) for x in args.corrections.split(",")]
    rows = []
    calibration = []
    for filename in args.inputs:
        payload = torch.load(filename, map_location="cpu", weights_only=False)
        keys = sorted(set(payload["blocks"]) & set(payload["queries"]))
        if args.max_pairs:
            keys = keys[:args.max_pairs]
        for key in keys:
            print(Path(filename).name, key, flush=True)
            counts, risk, mean_gap, trimmed_gap, one, variable, exact, qmean = evaluate(
                payload["blocks"][key], payload["queries"][key], args.extra,
                args.device, args.trim, args.tail,
            )
            common = {"input": filename, "sample": payload.get("sample_index"), "pair": str(key)}
            modes = {
                mode: tuple(aggregate(z, mode) for z in (one, variable, exact))
                for mode in ("mean", "max")
            }
            modes["qmean"] = qmean
            for mode, (l1, lr, target) in modes.items():
                uplift = lr - l1
                gap = target - lr
                for budget in budgets:
                    n = budget // payload.get("block_size", 8)
                    exact_mask = top_mask(target, n)
                    var_mask = top_mask(lr, n)
                    one_mask = top_mask(l1, n)
                    reference_correction = min(corrections, key=lambda x: abs(x - args.tail))
                    tail_corrected_mask = top_mask(
                        lr + reference_correction * risk[:, None], n
                    )
                    for method, score in (("one", l1), ("variable", lr)):
                        row = {**common, "gqa": mode, "budget": budget, "method": method,
                               **top_metrics(score, target, n)}
                        rows.append(row)
                    # Add the self-K risk as an estimated missing-mass correction.
                    for beta in corrections:
                        score = lr + beta * risk[:, None]
                        rows.append({**common, "gqa": mode, "budget": budget,
                                     "method": f"variable_plus_{beta:g}risk",
                                     **top_metrics(score, target, n)})
                    rows.append({**common, "gqa": mode, "budget": budget,
                                 "method": "variable_plus_mean_self_gap",
                                 **top_metrics(lr + mean_gap[:, None], target, n)})
                    rows.append({**common, "gqa": mode, "budget": budget,
                                 "method": "variable_plus_trimmed_self_gap",
                                 **top_metrics(lr + trimmed_gap[:, None], target, n)})
                    calibration.append({
                        **common, "gqa": mode, "budget": budget,
                        "mean_r": float(counts.float().mean()),
                        "global_r_gt1": float((counts > 1).float().mean()),
                        "global_r_gt2": float((counts > 2).float().mean()),
                        "exact_top_r_gt1": float((exact_mask * (counts > 1)[:, None]).sum(0).float().div(n).mean()),
                        "variable_top_r_gt1": float((var_mask * (counts > 1)[:, None]).sum(0).float().div(n).mean()),
                        "tail_corrected_top_r_gt1": float((tail_corrected_mask * (counts > 1)[:, None]).sum(0).float().div(n).mean()),
                        "one_top_r_gt1": float((one_mask * (counts > 1)[:, None]).sum(0).float().div(n).mean()),
                        "false_positive_r_gt1": float((((var_mask & ~exact_mask) * (counts > 1)[:, None]).sum() / (var_mask & ~exact_mask).sum().clamp_min(1))),
                        "false_negative_r1": float((((exact_mask & ~var_mask) * (counts == 1)[:, None]).sum() / (exact_mask & ~var_mask).sum().clamp_min(1))),
                        "tail_corrected_false_positive_r_gt1": float((((tail_corrected_mask & ~exact_mask) * (counts > 1)[:, None]).sum() / (tail_corrected_mask & ~exact_mask).sum().clamp_min(1))),
                        "tail_corrected_false_negative_r1": float((((exact_mask & ~tail_corrected_mask) * (counts == 1)[:, None]).sum() / (exact_mask & ~tail_corrected_mask).sum().clamp_min(1))),
                        "uplift_r1": float(uplift[counts == 1].mean()),
                        "uplift_r_gt1": float(uplift[counts > 1].mean()),
                        "exact_gap_r1": float(gap[counts == 1].mean()),
                        "exact_gap_r_gt1": float(gap[counts > 1].mean()),
                    })
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for name, data in (("ranking.csv", rows), ("calibration.csv", calibration)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    summary = {}
    for mode in ("mean", "max", "qmean"):
        for budget in budgets:
            for method in sorted(set(r["method"] for r in rows)):
                z = [r for r in rows if r["gqa"] == mode and r["budget"] == budget and r["method"] == method]
                summary[f"{mode}/B{budget}/{method}"] = {
                    k: sum(x[k] for x in z) / len(z) for k in ("overlap", "mass_recall", "best_hit")
                }
    csum = {}
    for mode in ("mean", "max", "qmean"):
        for budget in budgets:
            z = [r for r in calibration if r["gqa"] == mode and r["budget"] == budget]
            csum[f"{mode}/B{budget}"] = {
                k: sum(x[k] for x in z) / len(z) for k in z[0]
                if k not in {"input", "sample", "pair", "gqa", "budget"}
            }
    (out / "summary.json").write_text(json.dumps({"ranking": summary, "calibration": csum}, indent=2) + "\n")
    print(json.dumps({"ranking": summary, "calibration": csum}, indent=2))


if __name__ == "__main__":
    main()
