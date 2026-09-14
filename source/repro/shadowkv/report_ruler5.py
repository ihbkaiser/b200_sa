#!/usr/bin/env python
"""The RULER five-method table, merged across the three campaign machines.

The campaign is sharded, so no single machine holds the table.  Pulling the
cells to one host is not an option -- the m2 control plane carries commands and
status, never artifacts (PROJECT.md 3) -- so each machine emits a digest of a
few kB and the digests are merged here.

  # on each machine
  source repro/shadowkv/env_mX.sh
  $PY repro/shadowkv/report_ruler5.py --root $SHADOWKV_RESULTS_ROOT/ruler_five_20260913 \
      --emit-json /tmp/ruler5_mX.json

  # on m1, once the three are together
  $PY repro/shadowkv/report_ruler5.py --merge /tmp/ruler5_m1.json /tmp/ruler5_m2.json \
      /tmp/ruler5_m4.json

Reading rules, each of which has cost time when broken:

* a cell's score is the LAST line of its jsonl (the evaluator writes a running
  average, one line per sample), never the mean of the lines;
* a cell short of --samples is partial and never enters a comparison;
* the matrix is sparse while the campaign runs, so the only honest aggregate is
  PAIRWISE: for each pair of methods, the macro over the cells BOTH have.  A
  macro over "cells every method has" is empty until the slowest method lands,
  and a macro over each method's own cells compares different task mixes.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

# (tag in the cell name, short label).  Order matters: the first tag that
# matches wins, so put no tag that is a substring of another before it.
METHODS = [
    ("_full", "full"),
    ("adaptive_lse_stream", "ours"),
    ("quest_stream", "quest"),
    ("pariskv_author_common", "paris"),
    ("retroinfer_author_common", "retro"),
    ("shadowkv_cpu", "shadow"),
]
ORDER = ["full", "ours", "quest", "paris", "retro", "shadow"]
TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe",
    "qa_1", "qa_2",
]


def method_of(cell: str) -> str | None:
    for tag, name in METHODS:
        if tag == "_full":
            if cell.endswith("_full"):
                return "full"
        elif tag in cell:
            return name
    return None


def split_cell(cell: str):
    """-> (length, task, method) or None.

    The task is matched against the known RULER task list rather than parsed
    positionally: task names contain underscores, and so does every method tag.
    """
    parts = cell.split("_")
    if len(parts) < 3 or not parts[1].isdigit():
        return None
    length = int(parts[1])
    rest = "_".join(parts[2:])
    method = method_of(cell)
    if method is None:
        return None
    for task in sorted(TASKS, key=len, reverse=True):
        if rest.startswith(task + "_"):
            return length, task, method
    return None


def read_cell(path: str):
    """-> (n_lines, last avg_score).  A torn final line is dropped."""
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


def scan(root: str) -> dict:
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
                n, score = read_cell(os.path.join(mdir, name))
                out[name[: -len(".jsonl")]] = [n, score]
    return out


def fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "  -  "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=[],
                        help="a campaign root on this machine; repeatable")
    parser.add_argument("--merge", nargs="*", default=[],
                        help="digests written earlier by --emit-json")
    parser.add_argument("--emit-json", default=None)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    cells: dict[str, list] = {}
    for root in args.root:
        cells.update(scan(root))
    for path in args.merge:
        with open(path) as stream:
            # a cell that exists on two machines should not exist at all, but
            # if it does, keep the deeper copy rather than the last read
            for key, value in json.load(stream).items():
                if key not in cells or value[0] > cells[key][0]:
                    cells[key] = value

    if args.emit_json:
        with open(args.emit_json, "w") as stream:
            json.dump(cells, stream)
        print(f"{args.emit_json}: {len(cells)} cell(s)")
        return

    # (length, task, method) -> score, complete cells only
    grid: dict[tuple[int, str, str], float] = {}
    partial = []
    unparsed = []
    for cell, (n, score) in sorted(cells.items()):
        key = split_cell(cell)
        if key is None:
            unparsed.append(cell)
            continue
        if n < args.samples:
            partial.append((cell, n))
            continue
        if score is not None:
            grid[key] = score

    lengths = sorted({k[0] for k in grid})
    print(f"{len(cells)} cell(s) seen | {len(grid)} complete at {args.samples} "
          f"samples | {len(partial)} partial | {len(unparsed)} unparsed")

    for length in lengths:
        print(f"\n=== {length // 1024}K " + "=" * 52)
        header = "  ".join(f"{m:>6}" for m in ORDER)
        print(f"{'task':<18}{header}")
        for task in TASKS:
            row = [grid.get((length, task, m)) for m in ORDER]
            if not any(v is not None for v in row):
                continue
            print(f"{task:<18}" + "  ".join(f"{fmt(v):>6}" for v in row))

        # Pairwise, because the matrix is sparse: each pair is read over the
        # tasks BOTH methods finished, and the count is printed with it.
        print(f"\n  pairwise macro over shared tasks @{length // 1024}K")
        present = [m for m in ORDER if any(k[2] == m and k[0] == length for k in grid)]
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                shared = [t for t in TASKS
                          if (length, t, a) in grid and (length, t, b) in grid]
                if not shared:
                    continue
                ma = sum(grid[(length, t, a)] for t in shared) / len(shared)
                mb = sum(grid[(length, t, b)] for t in shared) / len(shared)
                mark = "  <-- " + (a if ma > mb else b) if abs(ma - mb) >= 0.01 else ""
                print(f"    {a:>6} {ma:.3f}  vs  {b:<6} {mb:.3f}   "
                      f"over {len(shared):>2} task(s){mark}")

    if partial:
        print(f"\n[partial, not in any table above]")
        for cell, n in partial:
            print(f"    {n:>4}/{args.samples}  {cell}")
    if unparsed:
        print(f"\n[unparsed cell names]")
        for cell in unparsed:
            print(f"    {cell}")


if __name__ == "__main__":
    main()
