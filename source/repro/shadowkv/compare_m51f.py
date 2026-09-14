#!/usr/bin/env python
"""The head-to-head the fixed-budget M51 was built for: same tokens, same tasks.

Unlike the ShadowKV/Quest comparison this needs no budget audit. m51f and streaming Quest
both attend exactly `budget` tokens per KV head, so the flag IS the pairing.
ShadowKV is shown at both readings, since its flag is not its budget.

  $PY repro/shadowkv/compare_m51f.py            # every length and budget present
  $PY repro/shadowkv/compare_m51f.py --per_task # where the differences sit
"""
import argparse, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import ROOT, MATCHED, load, paired, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--per_task", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cells = load(args.root)

    lens = sorted({k[1] for k in cells if k[0] == args.model})
    budgets = sorted({int(m.split("_")[1][1:]) for v in cells.values()
                      for m in v if m.startswith("m51f_b")})
    if not budgets:
        raise SystemExit("no m51f cells yet")

    for dl in lens:
        rows = []
        for b in budgets:
            m51f = next((m for v in cells.values() for m in v
                         if m.startswith(f"m51f_b{b}_")), None)
            if m51f is None:
                continue
            for other in (f"quest_stream_b{b}_p16_d2_max_x0_l32",
                          f"shadowkv_b{b}_r160_c8",
                          f"quest_stream_b{MATCHED.get(b, b)}_p16_d2_max_x0_l32"):
                keys = [k for k, v in cells.items()
                        if k[0] == args.model and k[1] == dl
                        and m51f in v and other in v
                        and len(v[m51f]) == len(v[other])]
                if not keys:
                    continue
                a = float(np.mean([cells[k][m51f].mean() for k in keys])) * 100
                c = float(np.mean([cells[k][other].mean() for k in keys])) * 100
                d, lo, hi = paired(cells, keys, m51f, other, rng)
                rows.append((b, other, len(keys), a, c, d * 100, lo * 100, hi * 100))
        if not rows:
            continue
        print(f"\n=== {args.model} @ {dl} | m51f vs baselines, RULER macro ===")
        print(f"{'budget':<8}{'baseline':<26}{'n':>4}{'m51f':>8}{'base':>8}"
              f"{'delta':>8}{'95% CI':>18}  verdict")
        print("-" * 96)
        for b, other, n, a, c, d, lo, hi in rows:
            print(f"{b:<8}{other:<26}{n:>4}{a:8.2f}{c:8.2f}{d:+8.2f}"
                  f"  [{lo:+6.2f},{hi:+6.2f}]  {verdict(lo, hi)}")

    if args.per_task:
        for b in budgets:
            m51f = next((m for v in cells.values() for m in v
                         if m.startswith(f"m51f_b{b}_")), None)
            q = f"quest_stream_b{b}_p16_d2_max_x0_l32"
            keys = [k for k, v in cells.items()
                    if k[0] == args.model and m51f in v and q in v]
            if not keys:
                continue
            print(f"\n--- per task, b{b}: m51f - quest ---")
            for k in sorted(keys, key=lambda k: (k[1], k[2])):
                d = (cells[k][m51f].mean() - cells[k][q].mean()) * 100
                print(f"  {k[1]:<7}{k[2]:<20}{cells[k][m51f].mean()*100:7.2f}"
                      f"{cells[k][q].mean()*100:8.2f}{d:+8.2f}")


if __name__ == "__main__":
    main()
