#!/usr/bin/env python3
"""Compare adaptive centroid routing with matched RULER baselines.

The adaptive campaign evaluates the first ``--samples`` rows of every cell.
Existing 96-row baselines are sliced to the identical prefix, so partial runs
can be inspected without mixing prompt sets.  Quest is reported both at the
same flag (b512) and at the attended-token-matched flag (b928); the latter is
the fair comparison because ShadowKV also retains 48x8 outlier tokens and a
small local window.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "vt", "cwe", "fwe",
    "qa_1", "qa_2",
)
MODELS = {
    "Qwen3-4B-Instruct-2507": (
        "qwen3",
        "qwen3-4b-instruct-2507",
    ),
    "Llama-3.2-3B-Instruct": (
        "llama32",
        "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    ),
}
METHODS = (
    "adaptive",
    "shadowkv",
    "quest_token_matched",
    "quest_same_flag",
    "full",
)


def last_scores(path: Path) -> list[float] | None:
    if not path.is_file():
        return None
    last = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    if not last or not last.get("correct"):
        return None
    return [float(value) for value in last["correct"]]


def cell_paths(adaptive_root: Path, baseline_root: Path, model: str, task: str):
    baseline_key, adaptive_key = MODELS[model]
    stem = f"{baseline_key}_16384_{task}"
    return {
        "adaptive": adaptive_root / adaptive_key / "ruler" /
        f"{task}_16384_shadowkv_centroid_lse_b512_r160_c8.jsonl",
        "shadowkv": baseline_root / baseline_key /
        f"{stem}_shadowkv_b512_r160_c8.jsonl",
        "quest_token_matched": baseline_root / baseline_key /
        f"{stem}_quest_b928_p16_d2_max.jsonl",
        "quest_same_flag": baseline_root / baseline_key /
        f"{stem}_quest_b512_p16_d2_max.jsonl",
        "full": baseline_root / baseline_key / f"{stem}_full.jsonl",
    }


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adaptive-root",
        type=Path,
        default=Path("paper_assets/certified_sparse/adaptive_cross_model_16k/results"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/storage/baonn/shadowkv_quest_20260827/cells"),
    )
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("paper_assets/certified_sparse/adaptive_cross_model_16k/results.csv"),
    )
    args = parser.parse_args()

    rows = []
    for model in MODELS:
        for task in TASKS:
            paths = cell_paths(args.adaptive_root, args.baseline_root, model, task)
            adaptive = last_scores(paths["adaptive"])
            n = min(len(adaptive), args.samples) if adaptive else 0
            scores: dict[str, float | None] = {}
            for method, path in paths.items():
                values = last_scores(path)
                scores[method] = (
                    sum(values[:n]) / n if values is not None and n else None
                )
            rows.append({
                "model": model,
                "task": task,
                "samples": n,
                "status": "done" if n == args.samples else ("partial" if n else "pending"),
                **scores,
            })

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "model", "task", "samples", "status", *METHODS,
        ))
        writer.writeheader()
        writer.writerows(rows)

    for model in MODELS:
        subset = [row for row in rows if row["model"] == model and row["samples"]]
        complete = sum(row["status"] == "done" for row in subset)
        print(f"\n{model}: {complete}/{len(TASKS)} complete, {len(subset)} observed")
        print(f"{'task':<20} {'n':>3} {'Adaptive':>9} {'ShadowKV':>9} "
              f"{'Quest928':>9} {'Quest512':>9} {'Full':>9}")
        for row in subset:
            print(
                f"{row['task']:<20} {row['samples']:>3} "
                f"{fmt(row['adaptive']):>9} {fmt(row['shadowkv']):>9} "
                f"{fmt(row['quest_token_matched']):>9} "
                f"{fmt(row['quest_same_flag']):>9} {fmt(row['full']):>9}"
            )
        complete_rows = [row for row in subset if row["status"] == "done"]
        if complete_rows:
            macro = {
                method: sum(row[method] for row in complete_rows) / len(complete_rows)
                for method in METHODS
            }
            print(
                f"{'complete-task macro':<20} {'':>3} "
                f"{fmt(macro['adaptive']):>9} {fmt(macro['shadowkv']):>9} "
                f"{fmt(macro['quest_token_matched']):>9} "
                f"{fmt(macro['quest_same_flag']):>9} {fmt(macro['full']):>9}"
            )

    print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
