#!/usr/bin/env python3
"""Aggregate angular-density trace screens without mixing their baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def trace_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(root.rglob("summary.json"))
            files.extend(path for path in root.glob("*.json") if path.is_file())
    return sorted(set(files))


def robust_baseline(method: str, methods: dict[str, dict]) -> str | None:
    if "_r1p25" in method:
        name = "ours_selfk_robusttrim_tailcvar_r1p25"
    elif "_r1p5" in method:
        name = "ours_selfk_robusttrim_tailcvar_r1p5"
    else:
        return None
    if method.endswith("_refine2x"):
        name += "_refine2x"
    return name if name in methods else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=512)
    args = parser.parse_args()

    records: list[dict] = []
    for path in trace_files(args.roots):
        payload = json.loads(path.read_text())
        methods = {
            row["method"]: row
            for row in payload["rows"]
            if row["budget"] == args.budget
        }
        if "ours" not in methods or "ours_refine_2x" not in methods:
            continue
        trace = path.parent.name if path.name == "summary.json" else path.stem
        for method, row in methods.items():
            if not method.startswith(("ours_density", "selfk_robusttrim")):
                continue
            reference = (
                "ours_refine_2x" if method.endswith("_refine2x") else "ours"
            )
            robust = robust_baseline(method, methods)
            score = 100.0 * row["total_mass_retained"]
            record = {
                "trace": trace,
                "method": method,
                "score": score,
                "delta_ours": score
                - 100.0 * methods[reference]["total_mass_retained"],
                "robust_baseline": robust,
                "delta_robust": (
                    score - 100.0 * methods[robust]["total_mass_retained"]
                    if robust else float("nan")
                ),
            }
            records.append(record)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("no compatible angular-density summaries found")
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "per_trace.csv", index=False)
    leaderboard = (
        frame.groupby("method", as_index=False)
        .agg(
            traces=("trace", "nunique"),
            score_mean=("score", "mean"),
            delta_ours_mean=("delta_ours", "mean"),
            delta_ours_min=("delta_ours", "min"),
            delta_ours_max=("delta_ours", "max"),
            delta_robust_mean=("delta_robust", "mean"),
            delta_robust_min=("delta_robust", "min"),
        )
        .sort_values("delta_robust_mean", ascending=False, na_position="last")
    )
    leaderboard.to_csv(args.output / "leaderboard.csv", index=False)
    print(leaderboard.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
