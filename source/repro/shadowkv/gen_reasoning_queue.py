#!/usr/bin/env python
"""Generate the reasoning matrix: MATH500, AIME25 and GPQA-diamond.

These benchmarks invert the long-context setting.  The prompt is a few hundred
tokens; the sequence that a sparse method has to index is the model's OWN
generation, which the streaming frame folds into the index every
STREAMING_UPDATE_INTERVAL tokens.  So a cell here measures decode-time
sparsity over self-generated context, and its cost is set by the generation
budget, not by the prompt.

Sampling is stochastic (temperature .6, top-p .9), so a single pass is a draw,
not a measurement -- most of all on AIME25, which has 30 problems, where one
problem is 3.3 points.  Seeds are therefore first-class: they ride in the task
key (``aime25-s2``), because the pool's environment is fixed for a whole queue
and only the nine queue fields vary cell by cell.  run_cell.sh unpacks the
suffix into SHADOWKV_GENERATION_SEED.

Everything here is a flag.  The defaults reproduce the September campaign's
shape -- datalen 32768, matched budget L/32 -- and are the thing to change
first when tuning:

  --tasks       which benchmarks
  --seeds       how many draws per benchmark (AIME needs the most)
  --methods     which methods, `full` included as the dense ceiling
  --datalen     the sequence scale the budget is matched against
  --budgets     one or more absolute budgets; overrides the matched L/32
"""

from __future__ import annotations

import argparse

# Per-benchmark seed counts: the smaller the benchmark, the more draws it takes
# before a difference between two methods means anything.  AIME25 is 30 items.
DEFAULT_SEEDS = {"aime25": 4, "math500": 1, "gpqa": 1}
# Published sizes, for the cost estimate the launcher prints.
SIZE = {"aime25": 30, "math500": 500, "gpqa": 198}
# The dataset's own max_new_tokens, which is what eval_acc reserves when the
# task key carries no -g override.
GEN = {"aime25": 32000, "math500": 4096, "gpqa": 16384}
# Deliberate overrides. GPQA ships at 16384 and runs at 32000 here, matching
# AIME: a cap the model can hit is a confound -- a wrong answer then means
# either the method lost the evidence or the answer was simply cut off, and the
# table cannot tell those apart. Raising it costs almost nothing, because the
# cost is what the model actually writes, not what it is allowed to. It does
# make these GPQA numbers incomparable with any GPQA run at 16384, which is why
# the cap goes into the task key and therefore into the cell name.
DEFAULT_GEN_OVERRIDE = {"gpqa": 32000}

METHODS = [
    "full",
    "adaptive_centroid_lse_streaming_prefix4_querymean",
    "quest_streaming",
    "pariskv_author_common",
    "retroinfer_author_common",
    "shadowkv_cpu",
]
# The dense ceiling and the two methods the comparison turns on come first, so
# a pool stopped early still leaves a readable row.
PRIORITY = ["full", "adaptive_centroid_lse_streaming_prefix4_querymean",
            "retroinfer_author_common", "pariskv_author_common"]
CHUNK = {"shadowkv_cpu": 4}
# ShadowKV cannot run these benchmarks, and the reason is its design, not our
# port. Its landmark index is the SVD of the PROMPT keys, taken once at prefill
# and frozen ("SV does not change after prefill"). A reasoning prompt is a few
# hundred tokens, far under sparse_budget, so the cache stays in dense_warmup
# and stays exact -- and when the generation crosses the budget there is no
# implemented path to build the basis, only
#   RuntimeError: ShadowKV exact warm-up reached sparse_budget;
#                 dynamic SVD transition is not implemented
# The method needs a long prefill to exist at all, so on a short-prompt
# benchmark the honest entry is N/A with that reason, not a number.
UNSUPPORTED = {
    "shadowkv": "its landmark basis is an SVD of the prompt, and these prompts "
                "are shorter than the budget -- see the note above",
    "shadowkv_cpu": "its landmark basis is an SVD of the prompt, and these "
                    "prompts are shorter than the budget -- see the note above",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--datalen", type=int, default=32768)
    parser.add_argument("--budgets", default="",
                        help="comma-separated absolute budgets; empty means "
                             "the single matched budget, datalen/32")
    parser.add_argument("--tasks", default="math500,aime25,gpqa")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--gen", default="",
                        help="task:max_new_tokens overrides, e.g. gpqa:32000; "
                             "empty keeps the built-in default overrides")
    parser.add_argument("--seeds", default="",
                        help="task:count pairs, e.g. aime25:4,math500:1; "
                             "unnamed tasks use the built-in default")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tasks = [x for x in args.tasks.split(",") if x]
    methods = [x for x in args.methods.split(",") if x]
    for method in methods:
        if method in UNSUPPORTED:
            raise SystemExit(
                f"'{method}' cannot run a short-prompt benchmark: "
                f"{UNSUPPORTED[method]}")
    budgets = ([int(x) for x in args.budgets.split(",") if x]
               or [args.datalen // 32])

    gen = dict(DEFAULT_GEN_OVERRIDE)
    for pair in args.gen.split(","):
        if pair:
            name, _, tokens = pair.partition(":")
            gen[name] = int(tokens)

    seeds = dict(DEFAULT_SEEDS)
    for pair in args.seeds.split(","):
        if pair:
            name, _, count = pair.partition(":")
            seeds[name] = int(count)
    for task in tasks:
        if task not in seeds:
            raise SystemExit(f"no seed count for '{task}' -- pass --seeds {task}:N")
        if task not in SIZE:
            raise SystemExit(f"unknown task '{task}' -- expected one of {sorted(SIZE)}")

    def line(task_key: str, method: str, budget: int) -> str:
        return (f"{args.model} {args.datalen} {task_key} {method} "
                f"{budget} 160 {CHUNK.get(method, 8)} 8 0")

    # Seed 0 of every (task, method) before seed 1 of anything: a campaign cut
    # short then has one complete draw of the whole matrix rather than four
    # draws of AIME and nothing else.
    # Budget before method before task, inside a seed: the first budget in the
    # list finishes as a complete table before the second one starts. Order the
    # list with the budget the comparison is anchored on first.
    max_seeds = max(seeds[t] for t in tasks)
    head = [m for m in methods if m in PRIORITY]
    tail = [m for m in methods if m not in PRIORITY]
    rows = []
    for seed in range(max_seeds):
        for budget in budgets:
            for method in head + tail:
                for task in tasks:
                    if seed < seeds[task]:
                        cap = f"-g{gen[task]}" if task in gen else ""
                        rows.append(line(f"{task}{cap}-s{seed}", method, budget))

    gen_tokens = (sum(SIZE[t] * gen.get(t, GEN[t]) * seeds[t] for t in tasks)
                  * len(methods) * len(budgets))
    with open(args.out, "w", encoding="utf-8") as stream:
        stream.write(
            f"# {len(rows)} cells | reasoning | datalen {args.datalen} "
            f"budgets {','.join(str(b) for b in budgets)} | seeds "
            + ",".join(f"{t}:{seeds[t]}" for t in tasks)
            + " | gen "
            + ",".join(f"{t}:{gen.get(t, GEN[t])}" for t in tasks) + "\n"
        )
        stream.write(
            f"# generation ceiling {gen_tokens/1e6:.1f}M tokens if nothing stops "
            f"early -- the smoke phase measures what it actually is\n"
        )
        stream.write("\n".join(rows) + "\n")
    print(f"{args.out}: {len(rows)} cells, "
          f"generation ceiling {gen_tokens/1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
