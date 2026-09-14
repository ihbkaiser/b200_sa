#!/usr/bin/env python
"""Generate the preregistered PQCache/MagicPIG/InfLLM RULER matrix.

The primary comparison fixes the routed/remote target to L/32: 1024 tokens at
32K and 2048 at 64K. With ``UPSTREAM_MATCHED_EXACT_REGIONS=1``, every method
also receives the same 32 exact prefix and 32 exact recent tokens. MagicPIG is
stochastic, so its realized remote sample count remains mandatory reporting.
"""

from __future__ import annotations

import argparse


TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe",
    "qa_1", "qa_2",
]
METHODS = [
    "pqcache_author_common", "magicpig_author_common",
    "infllm_author_common",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="llama32,qwen3")
    parser.add_argument("--lengths", default="32768,65536")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--chunk", type=int, default=8)
    args = parser.parse_args()

    models = [x for x in args.models.split(",") if x]
    lengths = [int(x) for x in args.lengths.split(",") if x]
    methods = [x for x in args.methods.split(",") if x]
    tasks = [x for x in args.tasks.split(",") if x]
    rows = []
    # Round-robin method/task/model so partial results are informative and a
    # single slow method does not postpone an entire table until the end.
    for length in lengths:
        budget = length // 32
        for task in tasks:
            for method in methods:
                for model in models:
                    rows.append(
                        f"{model} {length} {task} {method} "
                        f"{budget} 160 {args.chunk} {args.chunk} 0"
                    )
    print(
        f"# {len(rows)} cells | primary matched budget L/32 | "
        f"models={models} lengths={lengths} methods={methods}"
    )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
