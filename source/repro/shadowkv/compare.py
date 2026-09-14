#!/usr/bin/env python
"""
Read the campaign and answer the question it was run to answer: how do streaming Quest and
ShadowKV compare, and is any gap bigger than the noise.

  source repro/shadowkv/env_m1.sh
  $PY repro/shadowkv/compare.py                    # everything
  $PY repro/shadowkv/compare.py --split            # also per model and per length
  $PY repro/shadowkv/compare.py --per_task         # where the losses actually sit

Three rules this file exists to enforce:

  * Equal depth. A macro is only computed over the (model, length, task) cells
    that EVERY compared method has, and the count is printed next to it.
  * Paired, not marginal. All methods saw the same prompts under greedy
    decoding, so the informative statistic is the per-cell paired difference,
    not the distance between two independently-noisy macros.
  * A gap without a noise floor is not a result. Every comparison carries a
    95% CI from a cluster bootstrap over cells, and the verdict says "within
    noise" whenever that interval contains zero.
"""

import argparse
import collections
import json
import os
import re
import sys

import numpy as np

ROOT = os.environ.get("SHADOWKV_RESULTS_ROOT")

# ShadowKV always retains outlier_chunk*chunk_size plus a local window on top of
# the flag, so these are the pairs that actually attend the same number of
# tokens. Produced by budget_audit.py; see repro/shadowkv/README.md section 3.1.
MATCHED = {512: 896, 1024: 1408, 2048: 2432}


def parse_cell(name):
    """Split a generated cell name into model, length, task, and method."""
    parts = name.split("_")
    model, datalen, rest = parts[0], int(parts[1]), parts[2:]
    for i, tok in enumerate(rest):
        if tok in ("full", "quest", "shadowkv", "shadowkv_cpu", "m51", "m51f", "adaptive"):
            return model, datalen, "_".join(rest[:i]), "_".join(rest[i:])
    raise ValueError(f"cannot parse cell name: {name}")


def load(root):
    """(model, datalen, task) -> method -> per-sample score array."""
    cells = collections.defaultdict(dict)
    cells_dir = os.path.join(root, "cells")
    for model_key in sorted(os.listdir(cells_dir)):
        mdir = os.path.join(cells_dir, model_key)
        if not os.path.isdir(mdir):
            continue
        for fn in sorted(os.listdir(mdir)):
            if not fn.endswith(".jsonl"):
                continue
            name = fn[: -len(".jsonl")]
            model, datalen, task, method = parse_cell(name)
            last = None
            for line in open(os.path.join(mdir, fn)):
                if line.strip():
                    last = line
            if last is None:
                continue
            # `correct` on the final line is the cumulative per-sample list
            scores = json.loads(last).get("correct")
            if not scores:
                continue
            cells[(model, datalen, task)][method] = np.asarray(scores, dtype=float)
    return cells


def equal_depth(cells, methods):
    """Keys every one of `methods` has, plus the sample count they agree on."""
    keys = [k for k, v in cells.items() if all(m in v for m in methods)]
    dropped = []
    kept = []
    for k in sorted(keys):
        n = {len(cells[k][m]) for m in methods}
        (kept if len(n) == 1 else dropped).append(k)
    return kept, dropped


def macro(cells, keys, method):
    return float(np.mean([cells[k][method].mean() for k in keys]))


def paired(cells, keys, a, b, rng, iters=10000):
    """Paired per-cell difference a-b, with a cluster bootstrap CI over cells."""
    d = np.array([cells[k][a].mean() - cells[k][b].mean() for k in keys])
    idx = rng.integers(0, len(d), size=(iters, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return d.mean(), lo, hi


def verdict(lo, hi):
    return "within noise" if lo <= 0 <= hi else ("A better" if lo > 0 else "B better")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--split", action="store_true", help="also break down by model and length")
    ap.add_argument("--per_task", action="store_true", help="per-task deltas vs full attention")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.root:
        raise SystemExit("set SHADOWKV_RESULTS_ROOT or pass --root")

    rng = np.random.default_rng(args.seed)
    cells = load(args.root)
    methods = sorted({m for v in cells.values() for m in v})
    keys, dropped = equal_depth(cells, methods)

    print(f"=== {args.root} ===")
    print(f"{len(cells)} (model, length, task) cells | {len(methods)} method configs")
    print(f"equal-depth set: {len(keys)} cell(s) present in every method"
          + (f"  [{len(dropped)} dropped for disagreeing sample counts]" if dropped else ""))
    n_samples = {len(cells[keys[0]][m]) for m in methods}
    print(f"samples per cell: {n_samples.pop() if len(n_samples) == 1 else n_samples}")

    print("\n--- macro over the equal-depth set ---")
    for m in sorted(methods, key=lambda x: -macro(cells, keys, x)):
        print(f"  {m:<26} {macro(cells, keys, m)*100:6.2f}")

    # ---------------- the comparison the campaign was run for ----------------
    print("\n--- ShadowKV vs streaming Quest, the two readings ---")
    print("    Same --sparse_budget is NOT the same budget: ShadowKV attends")
    print("    budget + outlier(384) + local(~32). Both readings are shown.\n")
    hdr = f"  {'pairing':<34} {'shadowkv':>8} {'quest-s':>8} {'delta':>7}  {'95% CI':>17}  verdict"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for b, matched_b in MATCHED.items():
        sk = f"shadowkv_b{b}_r160_c8"
        for label, q in ((f"flag-matched   b{b} vs b{b}", f"quest_stream_b{b}_p16_d2_max_x0_l32"),
                         (f"token-matched  b{b} vs b{matched_b}", f"quest_stream_b{matched_b}_p16_d2_max_x0_l32")):
            if sk not in methods or q not in methods:
                continue
            d, lo, hi = paired(cells, keys, sk, q, rng)
            v = verdict(lo, hi).replace("A better", "ShadowKV").replace("B better", "Quest")
            print(f"  {label:<34} {macro(cells,keys,sk)*100:8.2f} {macro(cells,keys,q)*100:8.2f} "
                  f"{d*100:+7.2f}  [{lo*100:+6.2f},{hi*100:+6.2f}]  {v}")

    print("\n--- cost of sparsity: each method vs full attention ---")
    if "full" in methods:
        for m in sorted(methods):
            if m == "full":
                continue
            d, lo, hi = paired(cells, keys, m, "full", rng)
            tag = "within noise" if lo <= 0 <= hi else f"{d*100:+.2f} real"
            print(f"  {m:<26} {d*100:+6.2f}  [{lo*100:+6.2f},{hi*100:+6.2f}]  {tag}")

    # ---------------------------- breakdowns ---------------------------------
    if args.split:
        for axis, pick in (("model", lambda k: k[0]), ("length", lambda k: k[1])):
            print(f"\n--- by {axis} ---")
            for group in sorted({pick(k) for k in keys}, key=str):
                sub = [k for k in keys if pick(k) == group]
                print(f"  {axis}={group}  ({len(sub)} cells)")
                for b, matched_b in MATCHED.items():
                    sk, q = f"shadowkv_b{b}_r160_c8", f"quest_stream_b{matched_b}_p16_d2_max_x0_l32"
                    if sk not in methods or q not in methods:
                        continue
                    d, lo, hi = paired(cells, sub, sk, q, rng)
                    print(f"      token-matched b{b}: shadowkv {macro(cells,sub,sk)*100:5.2f} "
                          f"quest {macro(cells,sub,q)*100:5.2f}  delta {d*100:+5.2f} "
                          f"[{lo*100:+5.2f},{hi*100:+5.2f}]  {verdict(lo,hi).replace('A better','ShadowKV').replace('B better','Quest')}")

    if args.per_task and "full" in methods:
        print("\n--- per task: drop from full attention (macro over models+lengths) ---")
        tasks = sorted({k[2] for k in keys})
        show = [m for m in ("shadowkv_b1024_r160_c8", "quest_stream_b1408_p16_d2_max_x0_l32",
                            "quest_stream_b1024_p16_d2_max_x0_l32") if m in methods]
        print(f"  {'task':<18} {'full':>6} " + " ".join(f"{m.split('_b')[0][:8]+'_b'+m.split('_b')[1].split('_')[0]:>16}" for m in show))
        for t in tasks:
            sub = [k for k in keys if k[2] == t]
            row = f"  {t:<18} {macro(cells,sub,'full')*100:6.2f} "
            for m in show:
                row += f"{(macro(cells,sub,m)-macro(cells,sub,'full'))*100:+16.2f}"
            print(row)


if __name__ == "__main__":
    main()
