#!/usr/bin/env python3
"""Replay block-selection traces with a causal frequency reservoir."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--windows", default="4,8,16,32")
    ap.add_argument("--fractions", default="0.0625,0.125,0.25")
    ap.add_argument(
        "--layer-heads", default="",
        help="optional comma-separated layer:head pairs",
    )
    args = ap.parse_args()

    traces = torch.load(args.trace, map_location="cpu", weights_only=False)
    target = set(json.loads(args.summary.read_text())["evidence_blocks"])
    windows = [int(value) for value in args.windows.split(",")]
    fractions = [float(value) for value in args.fractions.split(",")]

    layer_heads = sorted({
        (int(entry["layer"]), head)
        for entry in traces
        for head in range(entry["selected_blocks"].shape[1])
    })
    if args.layer_heads:
        requested = {
            tuple(int(part) for part in value.split(":"))
            for value in args.layer_heads.split(",")
        }
        layer_heads = [pair for pair in layer_heads if pair in requested]
        if len(layer_heads) != len(requested):
            raise ValueError("one or more requested layer:head pairs are absent")
    sequences = {
        pair: [
            entry["selected_blocks"][0, pair[1]].tolist()
            for entry in traces if int(entry["layer"]) == pair[0]
        ]
        for pair in layer_heads
    }
    rows = []
    for window in windows:
        for fraction in fractions:
            for layer, head in layer_heads:
                history: deque[list[int]] = deque(maxlen=window)
                recalls = []
                for raw in sequences[(layer, head)]:
                    reserve = round(len(raw) * fraction)
                    frequency: Counter[int] = Counter()
                    recent_rank: dict[int, int] = {}
                    for previous in reversed(history):
                        for rank, block in enumerate(previous):
                            frequency[block] += 1
                            recent_rank.setdefault(block, rank)
                    keep = [
                        block for block, _ in sorted(
                            frequency.items(),
                            key=lambda item: (
                                -item[1], recent_rank[item[0]]
                            ),
                        )[:reserve]
                    ]
                    keep_set = set(keep)
                    selected = keep + [
                        block for block in raw if block not in keep_set
                    ][:len(raw) - len(keep)]
                    recalls.append(
                        sum(block in set(selected) for block in target)
                        / len(target)
                    )
                    history.append(raw)
                rows.append({
                    "window": window,
                    "fraction": fraction,
                    "layer": layer,
                    "head": head,
                    "recall_mean": sum(recalls) / len(recalls),
                    "recall_min": min(recalls),
                    "zero_steps": sum(value == 0 for value in recalls),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
