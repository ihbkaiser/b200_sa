#!/usr/bin/env python3
"""Report density-tail cells and their matched tail-CVaR controls."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TASKS = ("niah_multikey_3", "cwe", "qa_1", "qa_2")


def identity(name: str) -> tuple[int, str, int, float, str] | None:
    task = next((task for task in TASKS if f"_{task}_" in name), None)
    length_match = re.search(r"_(32768|131072)_", name)
    budget_match = re.search(r"_b(512|2048|4096)_", name)
    center_match = re.search(r"_x(0\.25|0\.5)_", name)
    if (
        task is None
        or length_match is None
        or budget_match is None
        or center_match is None
        or not any(
            tag in name
            for tag in ("allocdensity_tail_cvar", "alloctail_cvar")
        )
    ):
        return None
    centers = 1.0 + float(center_match.group(1))
    refinement = "rerank2b" if "_rtok" in name else "oneshot"
    return (
        int(length_match.group(1)), task, int(budget_match.group(1)),
        centers, refinement,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()

    rows = []
    for path in args.root.glob("cells/qwen3/*.jsonl"):
        key = identity(path.stem)
        if key is None:
            continue
        lines = [line for line in path.read_text().splitlines() if line]
        if not lines:
            continue
        payload = json.loads(lines[-1])
        rows.append((*key, len(lines), 100.0 * float(payload["avg_score"])))

    state = args.root / ".state"
    counts = {
        kind: len(list((state / kind).glob("*")))
        for kind in ("done", "running", "failed")
    }
    print(
        f"done={counts['done']} running={counts['running']} "
        f"failed={counts['failed']} rows={len(rows)}"
    )
    for length, task, budget, centers, refinement, samples, score in sorted(rows):
        print(
            f"{length // 1024:3d}K {task:18s} B={budget:<4d} "
            f"r={centers:<4g} {refinement:8s} "
            f"{samples:2d}/{args.samples} score={score:6.2f}"
        )


if __name__ == "__main__":
    main()
