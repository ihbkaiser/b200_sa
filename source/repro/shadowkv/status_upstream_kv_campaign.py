#!/usr/bin/env python
"""Show progress and partial accuracy for an upstream-KV campaign root."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def count_files(path: Path) -> int:
    return sum(1 for item in path.glob("*") if item.is_file()) if path.is_dir() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--state", default=None)
    args = parser.parse_args()
    root = args.root
    state = Path(args.state) if args.state else root / ".state"

    print(
        "progress "
        + " ".join(
            f"{name}={count_files(state / name)}"
            for name in ("done", "running", "failed")
        )
    )
    grouped: dict[tuple, list[tuple[float, int]]] = defaultdict(list)
    for stamp_path in (root / "cells").glob("*/*.stamp.json"):
        try:
            stamp = json.loads(stamp_path.read_text())
        except Exception:
            continue
        result_path = stamp_path.with_name(
            stamp_path.name.removesuffix(".stamp.json") + ".jsonl"
        )
        if not result_path.is_file() or result_path.stat().st_size == 0:
            continue
        try:
            rows = [json.loads(line) for line in result_path.read_text().splitlines() if line]
        except Exception:
            continue
        # Evaluator checkpoints are cumulative: line n contains correctness
        # for samples 1..n.  Only the final valid line is an independent
        # statistic; summing lines would triangularly over-count early items.
        row = rows[-1]
        correct = row.get("correct")
        if isinstance(correct, list) and correct:
            score_sum = sum(float(value) for value in correct)
            samples = len(correct)
        else:
            samples = len(row.get("prediction", [])) or 1
            score_sum = float(row["avg_score"]) * samples
        if not samples:
            continue
        model_key = stamp.get("cell", "model").split("_", 1)[0]
        key = (model_key,
               int(stamp["datalen"]), stamp["method"])
        grouped[key].append((score_sum / samples, samples))

    print(f"{'model':34} {'len':>6} {'method':28} {'cells':>5} {'samples':>7} {'avg':>7}")
    for (model, length, method), values in sorted(grouped.items()):
        total = sum(n for _, n in values)
        average = sum(score * n for score, n in values) / total if total else float("nan")
        print(f"{model[:34]:34} {length:6d} {method:28} {len(values):5d} {total:7d} {100*average:6.2f}")


if __name__ == "__main__":
    main()
