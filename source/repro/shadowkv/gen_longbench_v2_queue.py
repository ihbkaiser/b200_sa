#!/usr/bin/env python
"""Generate the five-method LongBench-v2 matrix at 128K.

Two length bands (short, medium) x the five methods that run their own fastest
variant on the shared frame.  The bands are the ones PROJECT.md 5 pins: 180 and
215 examples, selected by SHADOWKV_LONGBENCH_V2_LENGTH_FILTER inside run_cell,
so the queue line carries only the task name.

Queue order is a product decision.  ``full``, ours, RetroInfer and ParisKV are
emitted first across both bands -- the dense ceiling plus the three methods the
comparison turns on -- which is eight cells, exactly one claim on an eight-GPU
host.  The remaining methods follow band by band.

``--budgets`` sweeps the compression ratio.  ``full`` is emitted ONCE however
many budgets are asked for: dense attention does not read the budget, so a
second full cell at another budget is the same computation under a different
name, and it is the single most expensive cell here.
"""

from __future__ import annotations

import argparse

TASKS = ["longbench-v2-short", "longbench-v2-medium"]
# Queue order is a product decision: the first cells claimed are the ones a
# partial table needs.  full is the dense ceiling every sparse row is read
# against; ours and retroinfer are the two the comparison actually turns on.
PRIORITY = ["full", "adaptive_centroid_lse_streaming_prefix4_querymean",
            "retroinfer_author_common", "pariskv_author_common"]
METHODS = [
    "full",
    "adaptive_centroid_lse_streaming_prefix4_querymean",
    "quest_streaming",
    "pariskv_author_common",
    "retroinfer_author_common",
    "shadowkv_cpu",
]
CHUNK = {"shadowkv_cpu": 4}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--datalen", type=int, default=131072)
    parser.add_argument("--budgets", default="",
                        help="comma-separated budgets, primary first; empty "
                             "means the single matched budget, datalen/32")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    methods = [x for x in args.methods.split(",") if x]
    tasks = [x for x in args.tasks.split(",") if x]
    budgets = ([int(x) for x in args.budgets.split(",") if x]
               or [args.datalen // 32])

    def line(task: str, method: str, budget: int) -> str:
        return (
            f"{args.model} {args.datalen} {task} {method} "
            f"{budget} 160 {CHUNK.get(method, 8)} 8 0"
        )

    # Priority methods first, both bands each, then the rest band by band --
    # and the whole of that at one budget before the next one starts, so a
    # campaign cut short leaves one complete table rather than three partial.
    head = [m for m in PRIORITY if m in methods]
    rows = []
    seen = set()
    for budget in budgets:
        ordered = ([(t, m) for m in head for t in tasks]
                   + [(t, m) for t in tasks for m in methods if m not in head])
        for task, method in ordered:
            # dense does not read the budget: one full cell, not one per budget
            key = (task, method, None if method == "full" else budget)
            if key in seen:
                continue
            seen.add(key)
            rows.append(line(task, method, budget))
    with open(args.out, "w", encoding="utf-8") as stream:
        stream.write(
            f"# {len(rows)} cells | LongBench-v2 | budgets "
            f"{','.join(str(b) for b in budgets)} | model {args.model}\n"
        )
        stream.write("\n".join(rows) + "\n")
    print(f"{args.out}: {len(rows)} cells")


if __name__ == "__main__":
    main()
