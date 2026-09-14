#!/usr/bin/env python
"""
One command that prints the whole campaign: progress, failures, partial scores,
and an ETA built from measured per-cell cost.

  source repro/shadowkv/env_m1.sh
  $PY repro/shadowkv/status.py              # everything
  $PY repro/shadowkv/status.py --brief      # progress + comparison only

Reading rules this enforces, because getting them wrong has cost time before:
  * a cell's score is the LAST line of its jsonl (the evaluator writes a
    running average, one line per sample) -- never the mean of the lines;
  * a cell with fewer samples than its siblings is reported as partial and is
    excluded from the comparison table rather than silently averaged in;
  * the comparison is equal-depth: the macro for each method runs over the
    cells EVERY compared method has, and the count is printed.
"""

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_key import cell_key  # noqa: E402

ROOT = os.environ.get("SHADOWKV_RESULTS_ROOT")


def parse_cell(name):
    """cell name -> (model_key, datalen, task, method_tag). Exact, never substring."""
    parts = name.split("_")
    model_key, datalen = parts[0], parts[1]
    rest = parts[2:]
    for i, tok in enumerate(rest):
        if tok in ("full", "quest", "shadowkv", "m51", "adaptive") or tok.startswith(("quest_stream", "shadowkv")):
            return model_key, int(datalen), "_".join(rest[:i]), "_".join(rest[i:])
    return model_key, int(datalen), "_".join(rest), "?"


def read_cell(path):
    """-> (n_done, score) from a possibly-still-growing jsonl."""
    last, n = None, 0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                last = line
    except OSError:
        return 0, None
    if last is None:
        return 0, None
    try:
        return n, json.loads(last).get("avg_score")
    except json.JSONDecodeError:
        return n - 1, None            # a torn final line: do not count it


def collect(root):
    """<root>/cells/<model_key>/<cell>.jsonl -- the basename IS the cell key.

    run_cell.sh passes --cell_name from cell_key.py, so the jsonl, the stamp
    and the pool marker all carry the same string. Deriving a name here again
    would be a second definition, and two definitions of one cell drift.
    """
    cells = {}
    cells_dir = os.path.join(root, "cells")
    if not os.path.isdir(cells_dir):
        return cells
    for model_key in sorted(os.listdir(cells_dir)):
        mdir = os.path.join(cells_dir, model_key)
        if not os.path.isdir(mdir):
            continue
        for fn in sorted(os.listdir(mdir)):
            if not fn.endswith(".jsonl"):
                continue
            path = os.path.join(mdir, fn)
            n, score = read_cell(path)
            cells[fn[:-len(".jsonl")]] = dict(path=path, n=n, score=score,
                                              mtime=os.path.getmtime(path))
    return cells


def markers(state):
    out = {}
    for kind in ("done", "running", "failed"):
        d = os.path.join(state, kind)
        out[kind] = sorted(os.listdir(d)) if os.path.isdir(d) else []
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--state", default=None)
    ap.add_argument("--expected_samples", type=int, default=96)
    ap.add_argument("--brief", action="store_true")
    args = ap.parse_args()

    if not args.root:
        raise SystemExit("set SHADOWKV_RESULTS_ROOT or pass --root")
    state = args.state or os.path.join(args.root, ".state")

    print(f"=== {args.root} ===")
    print(f"    read at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    mk = markers(state)
    print(f"\n[markers] done {len(mk['done'])} | running {len(mk['running'])} | failed {len(mk['failed'])}")
    for f in mk["failed"]:
        print(f"    FAILED  {f}   log: {args.root}/_logs/{f}_gpu*.log")
    for r in mk["running"]:
        print(f"    running {r}")

    cells = collect(args.root)
    if not cells:
        print("\nno cells on disk yet")
        return

    complete = {k: v for k, v in cells.items() if v["n"] >= args.expected_samples}
    partial = {k: v for k, v in cells.items() if v["n"] < args.expected_samples}

    print(f"\n[cells] {len(cells)} on disk: {len(complete)} complete, {len(partial)} partial "
          f"(complete = {args.expected_samples} samples)")

    if partial and not args.brief:
        print("\n[partial]")
        for k, v in sorted(partial.items()):
            print(f"    {k:<64} {v['n']:>4}/{args.expected_samples}  score so far {v['score']}")

    # ---- equal-depth comparison -------------------------------------------
    by_method = collections.defaultdict(dict)     # method -> (model,len,task) -> score
    for name, v in complete.items():
        model_key, datalen, task, method = parse_cell(name)
        by_method[method][(model_key, datalen, task)] = v["score"]

    if len(by_method) >= 1:
        shared = None
        for m in by_method:
            keys = set(by_method[m])
            shared = keys if shared is None else (shared & keys)
        shared = shared or set()

        print(f"\n[comparison] equal-depth over {len(shared)} cell(s) present in all "
              f"{len(by_method)} method(s)")
        if shared:
            width = max(len(m) for m in by_method)
            for m in sorted(by_method):
                vals = [by_method[m][k] for k in shared if by_method[m][k] is not None]
                macro = sum(vals) / len(vals) if vals else float("nan")
                print(f"    {m:<{width}}  macro {macro:.4f}  over {len(vals)} cell(s)")
        for m in sorted(by_method):
            extra = set(by_method[m]) - shared
            if extra:
                print(f"    note: {m} has {len(extra)} cell(s) the others lack -- not counted above")

    if args.brief:
        return

    # ---- per-cell ----------------------------------------------------------
    print("\n[cells, newest first]")
    for name, v in sorted(cells.items(), key=lambda kv: -kv[1]["mtime"]):
        flag = " " if v["n"] >= args.expected_samples else "~"
        score = f"{v['score']:.4f}" if isinstance(v["score"], (int, float)) else str(v["score"])
        print(f"  {flag} {name:<64} {v['n']:>4}  {score}")

    # ---- ETA per cell type, not a global median ---------------------------
    # A cell's wall time is (last write to its jsonl) - (when the pool claimed
    # it). The claim time is the marker's mtime: the worker creates it under the
    # claim lock and `mv` preserves mtime, so it survives the move to done/.
    # Do NOT use getctime on the log -- on Linux that is inode change time and
    # it moves with every write, so it reads as zero elapsed.
    cost = collections.defaultdict(list)
    done_dir = os.path.join(state, "done")
    for cell in mk["done"]:
        if cell not in cells or cells[cell]["n"] <= 0:
            continue
        try:
            claimed = os.path.getmtime(os.path.join(done_dir, cell))
        except OSError:
            continue
        elapsed = cells[cell]["mtime"] - claimed
        if elapsed <= 0:
            continue
        _, datalen, _, method = parse_cell(cell)
        # bucket by method family, not by budget: budget barely moves the cost
        family = method.split("_b")[0]
        cost[(datalen, family)].append((elapsed, cells[cell]["n"]))

    if cost:
        print("\n[measured cost per cell, from claim to last write]")
        per_type = {}
        for (datalen, family), xs in sorted(cost.items()):
            secs = sorted(e for e, _ in xs)
            median = secs[len(secs) // 2]
            per_type[(datalen, family)] = median
            print(f"    {family:<14} @{datalen:<7} median {median/60:6.1f} min  "
                  f"over {len(secs)} cell(s)")

        # remaining work, priced by type; fall back to the slowest known type
        remaining = collections.Counter()
        queue = os.path.join(args.root, "queue.txt")
        if os.path.isfile(queue):
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            todo = 0
            for line in open(queue):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                try:
                    key = cell_key(*parts[:9], *(parts[9:10] or []))
                except SystemExit:
                    continue
                if key in mk["done"]:
                    continue
                todo += 1
                _, datalen, _, method = parse_cell(key)
                remaining[(datalen, method.split("_b")[0])] += 1

            worst = max(per_type.values())
            gpu_seconds = sum(n * per_type.get(k, worst) for k, n in remaining.items())
            unpriced = sum(n for k, n in remaining.items() if k not in per_type)
            workers = max(1, len(open(os.path.join(state, "gpus.txt")).read().split()))
            print(f"\n[ETA] {todo} cell(s) left, {gpu_seconds/3600:.1f} GPU-hours, "
                  f"{gpu_seconds/3600/workers:.1f} h wall-clock on {workers} worker(s)")
            if unpriced:
                pct = 100 * unpriced / max(1, todo)
                print(f"    {unpriced} of them ({pct:.0f}%) have no measured cost for their "
                      f"(length, method) yet and are priced at the slowest type measured SO FAR.")
                print(f"    That is not a bound in either direction: if the unmeasured types are "
                      f"slower -- and on the smoke, shadowkv ran ~5x a full-attention cell, and "
                      f"16K costs more per sample than 8K -- this number will rise as they land.")
                print(f"    Re-read it once every (length, method) family has completed at least "
                      f"one cell.")


if __name__ == "__main__":
    main()
