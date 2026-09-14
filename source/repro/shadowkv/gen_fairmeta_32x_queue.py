#!/usr/bin/env python
"""Generate a RULER metadata-matched 32x campaign.

The sparse budget is context/32.  Ours uses block 8, Quest uses two vectors
per page of 8, and ShadowKV uses one vector per chunk of 4.  ParisKV uses the
authors' SRHT/collision/radix/RaBitQ router inside the common model-forward
and evaluation path, at the same final token budget.
"""

import argparse

from gen_queue import RULER_TASKS


METHODS = (
    # method, rank, chunk, page, dense
    ("adaptive_centroid_lse_streaming_prefix4_querymean", 160, 8, 8, 0),
    ("quest_streaming", 160, 8, 8, 0),
    ("shadowkv_cpu", 160, 4, 8, 0),
    ("pariskv_author_common", 160, 8, 8, 0),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("qwen3", "llama32"), default="qwen3")
    parser.add_argument(
        "--datalens",
        default="32768,65536",
        help="comma-separated context lengths; every length must be divisible by 32",
    )
    parser.add_argument(
        "--budgets",
        default="",
        help=(
            "optional comma-separated sparse token budgets used at every length; "
            "the default is context/32"
        ),
    )
    parser.add_argument(
        "--include-full",
        action="store_true",
        help="append the dense ceiling to the four sparse methods",
    )
    parser.add_argument(
        "--exclude-ours",
        action="store_true",
        help="generate only baselines (useful while an ours-specific gate runs)",
    )
    args = parser.parse_args()
    methods = METHODS
    if args.exclude_ours:
        methods = tuple(row for row in methods if not row[0].startswith("adaptive_"))
    datalens = tuple(int(value) for value in args.datalens.split(",") if value)
    if not datalens or any(datalen <= 0 or datalen % 32 for datalen in datalens):
        parser.error("--datalens must contain positive multiples of 32")
    explicit_budgets = tuple(
        int(value) for value in args.budgets.split(",") if value
    )
    if any(budget <= 0 for budget in explicit_budgets):
        parser.error("--budgets must contain positive integers")

    rows = []
    for datalen in datalens:
        budgets = explicit_budgets or (datalen // 32,)
        for task in RULER_TASKS:
            for budget in budgets:
                for method, rank, chunk, page, dense in methods:
                    rows.append(
                        f"{args.model} {datalen} {task} {method} {budget} "
                        f"{rank} {chunk} {page} {dense}"
                    )
            # Dense output is budget-independent.  Emit it once rather than
            # duplicating the same cell for every sparse budget.
            if args.include_full:
                rows.append(
                    f"{args.model} {datalen} {task} full {budgets[0]} 160 8 8 0"
                )
    print(
        f"# {len(rows)} cells | {args.model} | 13 RULER tasks x 100 samples | "
        f"lengths={datalens}, budgets={explicit_budgets or 'context/32'} | "
        f"ours-c8, quest-p8, shadow-c4 | "
        f"full={'yes' if args.include_full else 'no'}"
    )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
