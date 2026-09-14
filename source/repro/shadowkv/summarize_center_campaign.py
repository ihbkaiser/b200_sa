#!/usr/bin/env python3
"""Aggregate independent block-8 center campaign trace summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    paths = sorted(args.root.glob("**/summary.csv"))
    if not paths:
        raise SystemExit(f"no summary.csv below {args.root}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["case"] = path.parent.name
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    metric = [
        "total_mass_retained",
        "candidate_mass_coverage",
        "target_attention_mass_coverage",
        "source_attention_mass_coverage",
        "mean_components",
        "fraction_blocks_r_ge_3",
    ]
    aggregate = raw.groupby(["budget", "method"], as_index=False).agg(
        cases=("case", "nunique"),
        **{name: (name, "mean") for name in metric},
    )
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        aggregate.to_csv(args.csv, index=False)

    expected = raw.case.nunique()
    complete = aggregate[aggregate.cases == expected].copy()
    references = complete[complete.method.isin([
        "ours", "pariskv", "exact_block8_totalmass", "exact_token_totalmass"
    ])]
    print(f"cases={expected} summaries={len(paths)} methods={len(complete)}")
    print("\nREFERENCES")
    print(references[["method", *metric[:4]]].to_string(index=False))

    for marker in ("r1p25", "r1p5", "rdlambda", "adaptive"):
        selected = complete[
            complete.method.str.contains("kproxy", regex=False)
            & complete.method.str.contains(marker, regex=False)
        ].sort_values("total_mass_retained", ascending=False)
        print(f"\n{marker.upper()}")
        print(selected[["method", *metric]].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
