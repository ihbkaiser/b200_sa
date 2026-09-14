#!/usr/bin/env python
"""
Emit a queue file: one cell per line, in the order the pool should claim them.

  $PY repro/shadowkv/gen_queue.py --models llama32,qwen3 --datalens 8192,16384 \
      --methods full,quest_streaming,shadowkv --budgets 1024,2048 > queue.txt

Line format (also the argument list run_cell.sh takes):
    model_key datalen task method budget rank chunk page dense

Ordering is a product decision, not a detail: --order cheap-first returns whole
tables sooner, --order expensive-first finishes the wall-clock sooner but keeps
every table incomplete until late. Default is cheap-first.
"""

import argparse
import sys

RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama32,qwen3")
    ap.add_argument("--datalens", default="8192,16384")
    ap.add_argument("--methods", default="full,quest_streaming,shadowkv")
    ap.add_argument("--budgets", default="1024,2048",
                    help="budgets for every sparse method unless overridden below")
    # Budgets are per-method on purpose. The same flag value does not buy the
    # same number of attended tokens: ShadowKV adds scaled outlier blocks plus
    # a local window on top.  The outlier count follows the paper's 0.293%
    # rate (see outlier_policy.py), so the offset depends on context length.
    # Run budget_audit.py to get matched values for a concrete configuration.
    ap.add_argument("--quest_budgets", default=None,
                    help="override --budgets for quest_streaming")
    ap.add_argument("--shadowkv_budgets", default=None,
                    help="override --budgets for shadowkv")
    ap.add_argument("--tasks", default=",".join(RULER_TASKS))
    ap.add_argument("--rank", type=int, default=160)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--page", type=int, default=16)
    ap.add_argument("--dense", type=int, default=2)
    ap.add_argument("--order", choices=["cheap-first", "expensive-first"], default="cheap-first")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    datalens = [int(d) for d in args.datalens.split(",") if d]
    methods = [m for m in args.methods.split(",") if m]
    if "quest" in methods:
        raise SystemExit(
            "method 'quest' was removed; use 'quest_streaming'"
        )
    def budget_list(spec):
        return sorted({int(b) for b in spec.split(",") if b})

    budgets = budget_list(args.budgets)
    per_method = {
        "quest_streaming": budget_list(args.quest_budgets) if args.quest_budgets else budgets,
        "shadowkv": budget_list(args.shadowkv_budgets) if args.shadowkv_budgets else budgets,
        "shadowkv_cpu": budget_list(args.shadowkv_budgets) if args.shadowkv_budgets else budgets,
    }
    tasks = [t for t in args.tasks.split(",") if t]

    lines = []
    for model in models:
        for datalen in datalens:
            for method in methods:
                # full attention has no budget axis: emitting it once per budget
                # would run the identical cell N times under N names.
                cell_budgets = [0] if method == "full" else per_method.get(method, budgets)
                for budget in cell_budgets:
                    if method != "full" and budget >= datalen:
                        print(f"# skipped {model} {datalen} {method} b{budget}: "
                              f"budget >= datalen", file=sys.stderr)
                        continue
                    for task in tasks:
                        lines.append((datalen, f"{model} {datalen} {task} {method} "
                                               f"{budget} {args.rank} {args.chunk} {args.page} {args.dense}"))

    lines.sort(key=lambda x: x[0], reverse=(args.order == "expensive-first"))
    shown = {m: per_method.get(m, budgets) for m in methods if m != "full"}
    print(f"# {len(lines)} cells | models={models} datalens={datalens} "
          f"methods={methods} budgets={shown} order={args.order}")
    for _, line in lines:
        print(line)


if __name__ == "__main__":
    main()
