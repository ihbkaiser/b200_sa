#!/usr/bin/env python
"""
Accuracy against traffic: put M51 on the same axes as Quest and ShadowKV.

A single operating point cannot be compared to anything. Each method is swept
(Quest and ShadowKV over budget, M51 over coverage target) and plotted as a
curve of RULER accuracy against the fraction of the KV cache its decode reads.

  source repro/shadowkv/env_m1.sh
  $PY repro/shadowkv/compare_traffic.py --model qwen3

Traffic definition, identical across methods so the axes mean one thing:
the fraction of context TOKENS a decode step attends to, per KV head.

  full      1.0
  quest_streaming  (budget + exact prefix/recent/tail) / prefill
  shadowkv  (budget + outlier_chunk*chunk_size + local) / prefill
  m51       measured -- the GQA-union block fraction, since query heads of a
            group share a cache and a deployment pays the union

Byte traffic differs further between methods (ShadowKV reads rank-160 rows and
reconstructs K; M51 also scans its PQ summaries). The summary scan is reported
as its own column for M51; ShadowKV's landmark scan is not charged here, so
ShadowKV's number is, if anything, generous.
"""

import argparse
import collections
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import load, parse_cell, macro, paired  # noqa: E402

def token_frac(method, datalen):
    if method == "full":
        return 1.0
    if method.startswith("quest_stream_b"):
        budget = int(method.split("_b")[1].split("_")[0])
        prefix = int(re.search(r"_x(\d+)", method).group(1))
        recent = int(re.search(r"_l(\d+)", method).group(1))
        return (budget + prefix + recent) / datalen
    if method.startswith("shadowkv_b"):
        budget = int(method.split("_b")[1].split("_")[0])
        outlier_match = re.search(r"_o(\d+)(?:_|$)", method)
        chunk_match = re.search(r"_c(\d+)(?:_|$)", method)
        # No _o suffix means a historical result from the fixed-48 campaign.
        outliers = int(outlier_match.group(1)) if outlier_match else 48
        chunk = int(chunk_match.group(1)) if chunk_match else 8
        return (budget + outliers * chunk + 32) / datalen
    return None                     # m51: measured, filled in from the traffic log


def load_traffic(root, model):
    """(datalen, coverage) -> measured fractions, averaged over cells."""
    path = os.path.join(root, "_traffic", f"{model}.jsonl")
    agg = collections.defaultdict(list)
    if not os.path.isfile(path):
        return {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        agg[(int(r["datalen"]), float(r["coverage"]))].append(r)
    out = {}
    for k, rows in agg.items():
        out[k] = {c: float(np.mean([r[c] for r in rows if c in r]))
                  for c in ("per_query_block_frac", "group_union_block_frac",
                            "summary_scan_frac", "resident_summary_frac",
                            "total_traffic_frac", "prefill_meta_s")
                  if any(c in r for r in rows)}
        out[k]["n_cells"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("SHADOWKV_RESULTS_ROOT"))
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cells = load(args.root)
    traffic = load_traffic(args.root, args.model)
    rng = np.random.default_rng(args.seed)

    lengths = sorted({k[1] for k in cells if k[0] == args.model})
    for L in lengths:
        keys_all = [k for k in cells if k[0] == args.model and k[1] == L]
        if not keys_all:
            continue
        methods = sorted(set.intersection(*[set(cells[k]) for k in keys_all])) \
            if keys_all else []
        keys = [k for k in keys_all if all(m in cells[k] for m in methods)]
        if not methods or not keys:
            continue

        print(f"\n=== {args.model} @ {L//1024}K  "
              f"({len(keys)} tasks present in all {len(methods)} configs) ===")
        print(f"  {'config':<28} {'RULER':>7} {'vs full':>9} {'tokens read':>12} "
              f"{'summary':>8} {'total':>7}")
        print("  " + "-" * 78)

        rows = []
        for m in methods:
            acc = macro(cells, keys, m) * 100
            if m == "full":
                d = 0.0
                tf, scan, tot = 1.0, 0.0, 1.0
            else:
                d = paired(cells, keys, m, "full", rng)[0] * 100 if "full" in methods else float("nan")
                if m.startswith("m51"):
                    cov = float(m.split("_c")[1].split("_")[0])
                    t = traffic.get((L, cov), {})
                    tf = t.get("group_union_block_frac", float("nan"))
                    scan = t.get("summary_scan_frac", float("nan"))
                    tot = t.get("total_traffic_frac", float("nan"))
                else:
                    tf = token_frac(m, L)
                    scan, tot = 0.0, tf
            rows.append((tot if tot == tot else 9.9, m, acc, d, tf, scan, tot))

        for _, m, acc, d, tf, scan, tot in sorted(rows):
            print(f"  {m:<28} {acc:7.2f} {d:+9.2f} {tf*100:11.2f}% "
                  f"{scan*100:7.2f}% {tot*100:6.2f}%")

        for cov in sorted({float(m.split('_c')[1].split('_')[0])
                           for m in methods if m.startswith('m51')}):
            t = traffic.get((L, cov), {})
            if t:
                print(f"    m51 c{cov}: per-query blocks {t.get('per_query_block_frac',0)*100:.2f}%, "
                      f"GQA-union {t.get('group_union_block_frac',0)*100:.2f}%, "
                      f"resident summaries {t.get('resident_summary_frac',0)*100:.2f}%, "
                      f"prefill metadata {t.get('prefill_meta_s',0):.1f}s over {t.get('n_cells',0)} cells")


if __name__ == "__main__":
    main()
