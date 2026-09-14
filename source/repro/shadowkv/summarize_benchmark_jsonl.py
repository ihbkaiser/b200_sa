#!/usr/bin/env python3
"""Summarize ShadowKV JSONL output, including benchmark subgroups."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()

    scores = []
    groups = defaultdict(lambda: defaultdict(list))
    with args.jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata = row.get("metadata", [{} for _ in row["correct"]])
            for score, meta in zip(row["correct"], metadata):
                value = float(score)
                scores.append(value)
                for key in ("difficulty", "length", "domain", "subject", "level"):
                    if meta.get(key) is not None:
                        groups[key][str(meta[key])].append(value)

    if not scores:
        raise SystemExit("no scores found")
    print(f"overall\t{sum(scores) / len(scores):.6f}\t{len(scores)}")
    for key, values in groups.items():
        for name, bucket in sorted(values.items()):
            print(f"{key}={name}\t{sum(bucket) / len(bucket):.6f}\t{len(bucket)}")


if __name__ == "__main__":
    main()
