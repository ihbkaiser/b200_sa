#!/usr/bin/env python
"""Two tables for the LongBench-v2 campaign: live progress, and a fair table.

  source repro/shadowkv/env_m3.sh
  $PY repro/shadowkv/report_lbv2.py $SHADOWKV_RESULTS_ROOT/longbench_v2_20260914

The two tables answer different questions and must not be conflated:

* **live** shows every cell with the samples it has so far.  A score at 37 of
  215 is not the cell's score and is never comparable with another cell's score
  at a different depth -- the column exists to show progress, not standing.
* **fair** keeps only cells that finished the whole band, and only tasks where
  every method finished, so each row is read at the same depth.

The band sizes are fixed by the dataset (PROJECT.md 5), not by the pool's
NUM_SAMPLES: the generic status.py assumes 96 and would call a 37/215 cell
"37/96" and a finished 180-sample cell complete for the wrong reason.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

BAND_SIZE = {"longbench-v2-short": 180, "longbench-v2-medium": 215}
METHODS = [
    ("full", "full"),
    ("adaptive_lse_stream", "ours"),
    ("quest_stream", "quest"),
    ("pariskv_author_common", "paris"),
    ("retroinfer_author_common", "retro"),
    ("shadowkv_cpu", "shadow"),
]
ORDER = ["full", "ours", "quest", "paris", "retro", "shadow"]


def method_of(cell: str) -> str | None:
    for tag, name in METHODS:
        if tag == "full":
            if cell.endswith("_full"):
                return "full"
        elif tag in cell:
            return name
    return None


def budget_of(cell: str) -> int:
    """The `_b<N>` field every sparse method's name carries; 0 for full.

    The budget is a row of the table, not a variant of the method. Two budgets
    of one method collapsed into one row would average a 512-token run into a
    4096-token one and call the result the method.
    """
    for part in cell.split("_"):
        if part.startswith("b") and part[1:].isdigit():
            return int(part[1:])
    return 0


def band_of(cell: str) -> str | None:
    for band in BAND_SIZE:
        if f"_{band}_" in cell or cell.endswith(f"_{band}"):
            return band
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()

    # An empty table and a missing directory look identical once printed, and
    # they mean opposite things: one is "the campaign has not landed a cell
    # yet", the other is "you are reading the wrong path". Reading the wrong
    # path is what actually happened, from a stale SHADOWKV_RESULTS_ROOT.
    if not os.path.isdir(os.path.join(args.root, "cells")):
        raise SystemExit(
            f"no cells/ under {args.root} -- this root holds no campaign.\n"
            f"Check $SHADOWKV_RESULTS_ROOT: sourcing env_mX.sh does not "
            f"overwrite a value already exported in the shell."
        )
    root = args.root

    rows: dict[tuple[int, str, str], tuple[float, int]] = {}
    for path in sorted(glob.glob(f"{root}/cells/*/*.jsonl")):
        cell = os.path.basename(path)[:-6]
        band, method = band_of(cell), method_of(cell)
        if not band or not method:
            continue
        lines = [json.loads(l) for l in open(path) if l.strip()]
        if not lines:
            continue
        # running average: the score is the last line, never the mean of lines
        rows[(budget_of(cell), method, band)] = (lines[-1]["avg_score"], len(lines))

    label = {"longbench-v2-short": "short", "longbench-v2-medium": "medium"}
    bands = list(BAND_SIZE)
    width = max(len(m) for m in ORDER) + 2

    # Dense is budget-independent, so it is printed once above the sweep rather
    # than repeated in every block as if it were a different measurement.
    dense = {b: rows[(0, "full", b)] for b in bands if (0, "full", b) in rows}
    budgets = sorted({k[0] for k in rows if k[0]}, reverse=True)
    sparse = [m for m in ORDER if m != "full"]

    def cell_text(entry, band):
        if entry is None:
            return "·".rjust(16)
        score, n = entry
        done = n >= BAND_SIZE[band]
        return (f"{score:.3f}  xong" if done
                else f"{score:.3f}  {n}/{BAND_SIZE[band]}").rjust(16)

    print("LongBench-v2 @128K, qwen3")
    print(f"{'':{width}}" + "".join(
        f"{label[b] + ' (' + str(BAND_SIZE[b]) + ')':>16}" for b in bands))
    if dense:
        print(f"{'full':{width}}" + "".join(
            cell_text(dense.get(b), b) for b in bands))
    for budget in budgets:
        print(f"-- budget {budget}  (L/{131072 // budget})")
        for m in sparse:
            print(f"{m:{width}}" + "".join(
                cell_text(rows.get((budget, m, b)), b) for b in bands))
    print("\n  xong = đã chạy hết band, điểm đọc được")
    print("  n/N  = đang chạy, điểm CHƯA đọc được (khác độ sâu thì không so)")
    print("  ·    = chưa có ô")

    # ---- fair table: one budget at a time, only bands every method finished --
    complete = {k: v[0] for k, v in rows.items() if v[1] >= BAND_SIZE[k[2]]}
    dense_ok = {b: complete[(0, "full", b)] for b in bands
                if (0, "full", b) in complete}
    print()
    printed = False
    for budget in budgets:
        ready = [b for b in bands
                 if all((budget, m, b) in complete for m in sparse)]
        if not ready:
            missing = {b: [m for m in sparse if (budget, m, b) not in complete]
                       for b in bands}
            print(f"[budget {budget}] chưa band nào đủ cả {len(sparse)} method — "
                  + "; ".join(f"{label[b]} thiếu " + ", ".join(missing[b])
                              for b in bands if missing[b]))
            continue
        printed = True
        print(f"[bảng công bằng, budget {budget}] chỉ band mọi method đã chạy hết")
        header = "".join(f"{label[b]:>10}" for b in ready)
        print(f"{'':{width}}" + header + (f"{'macro':>10}" if len(ready) > 1 else ""))
        for m in (["full"] if dense_ok else []) + sparse:
            source = dense_ok if m == "full" else None
            vals = [source[b] if source else complete[(budget, m, b)]
                    for b in ready if (source or {}).get(b) is not None
                    or (budget, m, b) in complete]
            if len(vals) != len(ready):
                continue
            line = "".join(f"{v:10.3f}" for v in vals)
            if len(ready) > 1:
                line += f"{sum(vals)/len(vals):10.3f}"
            print(f"{m:{width}}" + line)
        print(f"    equal-depth trên {len(ready)}/{len(bands)} band")
    if not printed:
        print("  (chưa budget nào có band hoàn chỉnh)")


if __name__ == "__main__":
    main()
