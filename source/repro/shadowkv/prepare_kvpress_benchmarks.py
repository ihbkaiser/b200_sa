#!/usr/bin/env python3
"""Materialize the processed datasets used by current NVIDIA/KVPress.

This downloads benchmark artifacts through Hugging Face on each machine; it
does not copy datasets between hosts. GPQA is intentionally excluded because
current upstream KVPress does not register it and its source data is gated.
Use ``repro/gpqa/build_gpqa_dataset.py`` for the local GPQA-Diamond adapter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


SPECS = {
    "longbench-v2": {
        "source": "simonjegou/LongBench-v2",
        "dirname": "LongBench-v2",
        "rows": 503,
        "max_new_tokens": 16,
        "columns": {
            "context", "question", "answer_prefix", "answer", "max_new_tokens",
            "difficulty", "length", "domain", "sub_domain", "_id",
        },
    },
    "aime25": {
        "source": "alessiodevoto/aime25",
        "dirname": "AIME25",
        "rows": 30,
        "max_new_tokens": 32000,
        "columns": {
            "context", "question", "answer_prefix", "answer", "max_new_tokens",
        },
    },
    "math500": {
        "source": "alessiodevoto/math500",
        "dirname": "MATH500",
        "rows": 500,
        "max_new_tokens": 4096,
        "columns": {
            "context", "question", "answer_prefix", "answer", "max_new_tokens",
            "subject", "level", "unique_id",
        },
    },
}


def materialize(name: str, output_root: Path) -> None:
    spec = SPECS[name]
    dataset = load_dataset(spec["source"], split="test")
    missing = spec["columns"] - set(dataset.column_names)
    generations = set(dataset["max_new_tokens"])
    if len(dataset) != spec["rows"] or missing or generations != {spec["max_new_tokens"]}:
        raise RuntimeError(
            f"{name} source changed: rows={len(dataset)}, missing={sorted(missing)}, "
            f"max_new_tokens={sorted(generations)}"
        )

    destination = output_root / spec["dirname"]
    parquet = destination / "test-00000-of-00001.parquet"
    if parquet.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {parquet}")
    destination.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(parquet)
    print(f"READY {name}: {len(dataset)} rows -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--benchmark", choices=["all", *SPECS], default="all",
    )
    args = parser.parse_args()
    names = SPECS if args.benchmark == "all" else (args.benchmark,)
    for name in names:
        materialize(name, args.output_root)


if __name__ == "__main__":
    main()
