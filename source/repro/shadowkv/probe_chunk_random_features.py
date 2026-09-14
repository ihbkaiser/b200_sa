#!/usr/bin/env python3
"""Probe positive random features as a per-block softmax-mass router.

This consumes the raw post-RoPE K/Q tensors written by the block-geometry
diagnostics.  It compares exact block mass, a single centroid, and the
positive random-feature estimator at equal or larger metadata budgets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


def qtile(x: torch.Tensor, values=(0.5, 0.9, 0.99)) -> list[float]:
    return [float(v) for v in torch.quantile(x.float().flatten(), torch.tensor(values, device=x.device))]


def top_metrics(pred: torch.Tensor, truth: torch.Tensor, top_blocks: int) -> dict[str, float]:
    """Average set overlap and exact-mass recall over columns (decode steps)."""
    k = min(top_blocks, truth.shape[0])
    true_idx = truth.topk(k, dim=0).indices
    pred_idx = pred.topk(k, dim=0).indices
    # P is only ~4K, so dense indicators are both simple and cheap.
    true_mask = torch.zeros_like(truth, dtype=torch.bool).scatter_(0, true_idx, True)
    pred_mask = torch.zeros_like(truth, dtype=torch.bool).scatter_(0, pred_idx, True)
    overlap = (true_mask & pred_mask).sum(0).float() / k
    # Subtract max before exponentiating; the ratio is unchanged.
    shifted = (truth - truth.max(0, keepdim=True).values).exp()
    recall = (shifted * pred_mask).sum(0) / shifted.sum(0)
    best_hit = pred_mask.gather(0, truth.argmax(0, keepdim=True)).squeeze(0).float()
    return {
        "overlap_mean": float(overlap.mean()),
        "overlap_p10": float(torch.quantile(overlap, 0.1)),
        "mass_recall_mean": float(recall.mean()),
        "mass_recall_p10": float(torch.quantile(recall, 0.1)),
        "best_block_hit": float(best_hit.mean()),
    }


def aggregate(logz: torch.Tensor, mode: str) -> torch.Tensor:
    """[P,H,T] -> [P,T], preserving the mass scale across GQA heads."""
    if mode == "mean":
        return torch.logsumexp(logz, dim=1) - math.log(logz.shape[1])
    if mode == "max":
        return logz.max(dim=1).values
    raise ValueError(mode)


def make_omega(m: int, d: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(m, d, generator=gen, device=device, dtype=torch.float32)


@torch.inference_mode()
def evaluate_pair(
    blocks_cpu: torch.Tensor,
    queries_cpu: torch.Tensor,
    dimensions: list[int],
    seeds: list[int],
    budgets: list[int],
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    k = blocks_cpu.to(device=device, dtype=torch.float32)
    q = queries_cpu.to(device=device, dtype=torch.float32)
    p, c, d = k.shape
    h, t, qd = q.shape
    assert d == qd
    scale = math.sqrt(d)

    # Exact mass and centroid baseline.
    logits = (k.reshape(p * c, d) @ q.reshape(h * t, d).T / scale).reshape(p, c, h, t)
    exact = torch.logsumexp(logits, dim=1)
    centroid = math.log(c) + torch.einsum("pcd,htd->pht", k.mean(1, keepdim=True), q).squeeze(1) / scale

    rows: list[dict] = []
    errors: list[dict] = []
    for mode in ("mean", "max"):
        target = aggregate(exact, mode)
        base = aggregate(centroid, mode)
        for budget in budgets:
            rows.append({"method": "centroid", "M": d, "seed": -1, "gqa": mode,
                         "budget_tokens": budget, **top_metrics(base, target, budget // c)})

    # exp(q.k/sqrt(d)) = exp((q/d^.25).(k/d^.25)).
    fourth_root = d ** 0.25
    x = q.reshape(h * t, d) / fourth_root
    y = k.reshape(p * c, d) / fourth_root
    y_norm = 0.5 * y.square().sum(1, keepdim=True)
    x_norm = 0.5 * x.square().sum(1, keepdim=True)
    max_m = max(dimensions)

    for seed in seeds:
        omega = make_omega(max_m, d, seed, device)
        log_fk = y @ omega.T - y_norm                    # [P*C,M]
        log_u_all = torch.logsumexp(log_fk.reshape(p, c, max_m), dim=1)  # [P,M]
        log_fq_all = x @ omega.T - x_norm                # [H*T,M]
        del log_fk

        for m in dimensions:
            log_u = log_u_all[:, :m]
            log_fq = log_fq_all[:, :m]
            # Stable positive-feature GEMM. Column and row shifts cancel.
            col_shift = log_u.max(0).values
            a = (log_u - col_shift).exp()
            shifted_b = log_fq + col_shift
            row_shift = shifted_b.max(1, keepdim=True).values
            b = (shifted_b - row_shift).exp()
            dots = (a @ b.T).clamp_min_(torch.finfo(torch.float32).tiny)
            rf = (dots.log() + row_shift.T - math.log(m)).reshape(p, h, t)

            for mode in ("mean", "max"):
                target = aggregate(exact, mode)
                pred = aggregate(rf, mode)
                for budget in budgets:
                    rows.append({"method": "prf", "M": m, "seed": seed, "gqa": mode,
                                 "budget_tokens": budget, **top_metrics(pred, target, budget // c)})

                log_ratio = pred - target
                rank = target.argsort(dim=0, descending=True).argsort(dim=0)
                top1 = rank < max(1, math.ceil(0.01 * p))
                top10 = rank < max(1, math.ceil(0.10 * p))
                for region, mask in (("top1pct", top1), ("top10pct", top10), ("all", torch.ones_like(top1))):
                    z = log_ratio[mask]
                    errors.append({
                        "M": m, "seed": seed, "gqa": mode, "region": region,
                        "log_ratio_median": float(z.median()),
                        "abs_log_error_median": float(z.abs().median()),
                        "abs_log_error_p90": qtile(z.abs(), (0.9,))[0],
                    })
            del a, b, dots, rf
    return rows, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dimensions", default="32,64,128,256")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--budgets", default="512,2048")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="Limit layer/head pairs per input (0 means all).")
    args = ap.parse_args()

    dimensions = [int(x) for x in args.dimensions.split(",")]
    budgets = [int(x) for x in args.budgets.split(",")]
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    all_errors: list[dict] = []

    for input_name in args.inputs:
        payload = torch.load(input_name, map_location="cpu", weights_only=False)
        keys = sorted(set(payload["blocks"]) & set(payload["queries"]))
        if args.max_pairs:
            keys = keys[: args.max_pairs]
        for key in keys:
            print(f"[{Path(input_name).name}] {key}", flush=True)
            rows, errors = evaluate_pair(payload["blocks"][key], payload["queries"][key],
                                         dimensions, list(range(args.seeds)), budgets, device)
            common = {"input": str(input_name), "sample": payload.get("sample_index"), "pair": str(key)}
            all_rows.extend([{**common, **r} for r in rows])
            all_errors.extend([{**common, **r} for r in errors])

    for name, records in (("ranking.csv", all_rows), ("errors.csv", all_errors)):
        with (output / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)

    # Compact aggregate summary (mean over pairs/steps already reduced within pair).
    summary = {}
    for method in ("centroid", "prf"):
        for mode in ("mean", "max"):
            for budget in budgets:
                for m in ([128] if method == "centroid" else dimensions):
                    subset = [r for r in all_rows if r["method"] == method and r["gqa"] == mode
                              and r["budget_tokens"] == budget and r["M"] == m]
                    if not subset:
                        continue
                    label = f"{method}/M{m}/{mode}/B{budget}"
                    summary[label] = {k: sum(r[k] for r in subset) / len(subset)
                                      for k in ("overlap_mean", "mass_recall_mean", "best_block_hit")}
                    summary[label]["n_pair_seeds"] = len(subset)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
