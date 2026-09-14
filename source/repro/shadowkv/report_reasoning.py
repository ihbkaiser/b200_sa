#!/usr/bin/env python
"""The reasoning table -- MATH500 / AIME25 / GPQA -- and the cost projection.

  source repro/shadowkv/env_m3.sh
  $PY repro/shadowkv/report_reasoning.py $ROOT/smoke --cost   # before committing cards
  $PY repro/shadowkv/report_reasoning.py $ROOT/full           # the table

Two things this enforces, both of which have been got wrong before:

* a cell's score is the LAST line of its jsonl -- the evaluator writes a running
  average, one line per sample -- never the mean of the lines;
* seeds are draws of the same cell, so they are shown as mean and range across
  seeds, never pooled into one number and never silently averaged when one seed
  is short.  On AIME25 a single problem is 3.3 points and the range is usually
  the more informative half;
* the budget is part of the row, not part of the method.  Two budgets of one
  method are two rows: collapsing them would average a 512-token run into a
  1024-token one and call the result the method.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re

METHODS = [
    ("_full", "full"),
    ("adaptive_lse_stream", "ours"),
    ("quest_stream", "quest"),
    ("pariskv_author_common", "paris"),
    ("retroinfer_author_common", "retro"),
    ("shadowkv_cpu", "shadow"),
]
ORDER = ["full", "ours", "quest", "paris", "retro", "shadow"]
# The queue writes a method's long name; the cell name writes its tag. They are
# not the same string -- `adaptive_centroid_lse_streaming_prefix4_querymean`
# becomes `adaptive_lse_stream` -- so pricing a queue needs this map, not
# method_of().
QUEUE_METHOD = {
    "full": "full",
    "adaptive_centroid_lse_streaming_prefix4_querymean": "ours",
    "quest_streaming": "quest",
    "pariskv_author_common": "paris",
    "retroinfer_author_common": "retro",
    "shadowkv_cpu": "shadow",
    "shadowkv": "shadow",
}
# Published sizes: what a complete cell must reach, whatever the pool asked for.
SIZE = {"math500": 500, "aime25": 30, "gpqa": 198}
TASKS = ["math500", "aime25", "gpqa"]


def method_of(cell: str) -> str | None:
    for tag, name in METHODS:
        if tag == "_full":
            if cell.endswith("_full"):
                return "full"
        elif tag in cell:
            return name
    return None


def budget_of(cell: str) -> int:
    """The `_b<N>` field every sparse method's name carries; 0 for full."""
    for part in cell.split("_"):
        if part.startswith("b") and part[1:].isdigit():
            return int(part[1:])
    return 0


# <bench>[-g<max_new_tokens>]-s<seed>, as run_cell.sh writes it into the name.
TASK_KEY = re.compile(
    r"_(" + "|".join(TASKS) + r")(?:-g(\d+))?-s(\d+)(?:_|$)")


def split_cell(cell: str):
    """-> (task, seed, method, budget), or None if not a reasoning cell."""
    method = method_of(cell)
    if method is None:
        return None
    match = TASK_KEY.search(cell)
    if match is None:
        return None
    return match.group(1), int(match.group(3)), method, budget_of(cell)


def read_cell(path: str):
    last, n = None, 0
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if line:
                n += 1
                last = line
    if last is None:
        return 0, None
    try:
        return n, json.loads(last).get("avg_score")
    except json.JSONDecodeError:
        return n - 1, None


def scan(root: str):
    """-> {cell: (n, score, mtime)}"""
    out = {}
    cells_dir = os.path.join(root, "cells")
    if not os.path.isdir(cells_dir):
        raise SystemExit(
            f"no cells/ under {root} -- this root holds no campaign.\n"
            f"Check $SHADOWKV_RESULTS_ROOT: sourcing env_mX.sh does not "
            f"overwrite a value already exported in the shell.")
    for model_key in sorted(os.listdir(cells_dir)):
        mdir = os.path.join(cells_dir, model_key)
        if not os.path.isdir(mdir):
            continue
        for name in sorted(os.listdir(mdir)):
            if name.endswith(".jsonl"):
                path = os.path.join(mdir, name)
                n, score = read_cell(path)
                out[name[: -len(".jsonl")]] = (n, score, os.path.getmtime(path))
    return out


def table(cells, root):
    # (task, method, budget) -> {seed: (n, score)}
    grid = collections.defaultdict(dict)
    for cell, (n, score, _) in cells.items():
        key = split_cell(cell)
        if key is None:
            continue
        task, seed, method, budget = key
        grid[(task, method, budget)][seed] = (n, score)

    tasks = [t for t in TASKS if any(k[0] == t for k in grid)]
    budgets = sorted({k[2] for k in grid}, reverse=True)
    rows = [(m, b) for b in budgets for m in ORDER
            if any(k[1] == m and k[2] == b for k in grid)]

    print(f"{'':<15}" + "".join(f"{t:>22}" for t in tasks))
    print(f"{'':<15}" + "".join(f"{'(n=' + str(SIZE[t]) + ')':>22}" for t in tasks))
    last_budget = None
    for method, budget in rows:
        if budget != last_budget:
            print(f"-- budget {budget or 'dense'}")
            last_budget = budget
        row = f"{method:<15}"
        for task in tasks:
            draws = grid.get((task, method, budget), {})
            full = {s: v for s, v in draws.items() if v[0] >= SIZE[task]}
            if not full:
                depth = "/".join(str(v[0]) for _, v in sorted(draws.items()))
                row += f"{('~' + depth) if depth else '-':>22}"
                continue
            scores = [v[1] for _, v in sorted(full.items()) if v[1] is not None]
            mean = sum(scores) / len(scores)
            if len(scores) == 1:
                cell = f"{mean:.3f}"
            else:
                cell = f"{mean:.3f} [{min(scores):.3f}-{max(scores):.3f}]"
            short = len(draws) - len(full)
            if short:
                cell += f" +{short}~"
            row += f"{cell:>22}  x{len(scores)}"[:22].rjust(22)
        print(row)
    print("\n  x<k> = seeds averaged | [lo-hi] = range across seeds | "
          "~<n> = partial, not scored | +k~ = k seeds still short")


def cost(cells, root, samples_seen, queue_path):
    """Per-sample wall time from the smoke phase, projected onto a real queue.

    A cell's wall time is (last write to its jsonl) - (the pool's claim), and
    the claim time is the marker's mtime: the worker creates it under the claim
    lock and `mv` preserves mtime through done/.  Loading the model sits inside
    that window, so at small sample counts this is an UPPER bound per sample.

    Cost is bucketed by (task, method) and not by budget.  What these cells
    spend is generation, and the budget moves the per-step cost by a few
    percent while the number of steps is set by the benchmark -- pricing each
    budget separately would just halve the sample behind every number.
    """
    done_dir = os.path.join(root, ".state", "done")
    if not os.path.isdir(done_dir):
        raise SystemExit(f"no finished cells under {root}/.state/done")

    seconds = collections.defaultdict(list)
    for cell in sorted(os.listdir(done_dir)):
        key = split_cell(cell)
        if key is None or cell not in cells:
            continue
        n, _, mtime = cells[cell]
        if n <= 0:
            continue
        elapsed = mtime - os.path.getmtime(os.path.join(done_dir, cell))
        if elapsed > 0:
            task, _, method, _budget = key
            seconds[(task, method)].append(elapsed / n)
    if not seconds:
        raise SystemExit("no finished smoke cell to price yet")
    per = {k: sum(v) / len(v) for k, v in seconds.items()}

    print(f"[measured] seconds per sample, {samples_seen} sample(s) per cell, "
          f"model load included -- so an upper bound\n")
    tasks = sorted({k[0] for k in per})
    methods = [m for m in ORDER if any(k[1] == m for k in per)]
    print(f"{'':<8}" + "".join(f"{t:>12}" for t in tasks))
    for method in methods:
        row = f"{method:<8}"
        for task in tasks:
            value = per.get((task, method))
            row += f"{value:>12.1f}" if value else f"{'-':>12}"
        print(row)

    if not queue_path:
        print("\n  pass --queue <full-phase queue> to price the run you mean to launch")
        return
    slowest = max(per.values())
    hours, unpriced = 0.0, 0
    lines = 0
    for line in open(queue_path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        task = re.sub(r"(-g\d+)?-s\d+$", "", fields[2])
        method = QUEUE_METHOD.get(fields[3])
        if task not in SIZE or method is None:
            raise SystemExit(f"cannot price queue line: {line}")
        rate = per.get((task, method))
        if rate is None:
            rate, unpriced = slowest, unpriced + 1
        hours += rate * SIZE[task] / 3600
        lines += 1
    print(f"\n[projection] {queue_path}")
    print(f"  {lines} cell(s), {hours:.0f} GPU-hours, "
          f"{hours / 8:.0f} h wall-clock on 8 cards")
    if unpriced:
        print(f"  {unpriced} of them have no measured (task, method) yet and are "
              f"priced at the slowest one measured so far -- not a bound either way")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--cost", action="store_true",
                        help="price a run from this (smoke) root")
    parser.add_argument("--queue", default=None,
                        help="with --cost: the queue file to price")
    args = parser.parse_args()

    cells = scan(args.root)
    reasoning = {c: v for c, v in cells.items() if split_cell(c)}
    print(f"=== {args.root} ===")
    print(f"{len(reasoning)} reasoning cell(s) of {len(cells)} on disk\n")
    if args.cost:
        depths = {v[0] for v in reasoning.values()} or {0}
        cost(cells, args.root, max(depths), args.queue)
    else:
        table(reasoning, args.root)


if __name__ == "__main__":
    main()
