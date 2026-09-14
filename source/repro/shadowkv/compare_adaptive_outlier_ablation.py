#!/usr/bin/env python3
"""Equal-prefix comparison of our adaptive router with 48 vs zero outliers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
MODELS = {
    "Qwen3-4B-Instruct-2507": (
        "qwen3", "qwen3-4b-instruct-2507",
    ),
    "Llama-3.2-3B-Instruct": (
        "llama32", "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    ),
}


def scores(path: Path) -> list[float]:
    if not path.is_file():
        return []
    last = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
    return [float(x) for x in (last or {}).get("correct", [])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-root", type=Path,
        default=Path("/storage/baonn/adaptive_centroid_no_outlier_20260904_ruler16k/cells"),
    )
    parser.add_argument(
        "--old-root", type=Path,
        default=Path("paper_assets/certified_sparse/adaptive_cross_model_16k/results"),
    )
    parser.add_argument("--max-samples", type=int, default=24)
    args = parser.parse_args()

    for label, (new_key, old_key) in MODELS.items():
        print(f"\n{label}")
        print(f"{'task':<20} {'n':>3} {'ours-o0':>9} {'ours-o48':>10} {'delta':>8}")
        paired = []
        for task in TASKS:
            new = scores(
                args.new_root / new_key /
                f"{new_key}_16384_{task}_adaptive_lse_b512_r160_c8_x0.25_t1_o0.jsonl"
            )
            old = scores(
                args.old_root / old_key / "ruler" /
                f"{task}_16384_shadowkv_centroid_lse_b512_r160_c8.jsonl"
            )
            n = min(len(new), len(old), args.max_samples)
            if not n:
                print(f"{task:<20} {0:>3} {'-':>9} {'-':>10} {'-':>8}")
                continue
            a = sum(new[:n]) / n
            b = sum(old[:n]) / n
            paired.append((a, b))
            print(f"{task:<20} {n:>3} {100*a:>9.2f} {100*b:>10.2f} {100*(a-b):>+8.2f}")
        if paired:
            new_macro = sum(a for a, _ in paired) / len(paired)
            old_macro = sum(b for _, b in paired) / len(paired)
            print(
                f"{'equal-depth macro':<20} {len(paired):>3} "
                f"{100*new_macro:>9.2f} {100*old_macro:>10.2f} "
                f"{100*(new_macro-old_macro):>+8.2f}"
            )


if __name__ == "__main__":
    main()
