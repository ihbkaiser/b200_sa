#!/usr/bin/env python
"""Render the M51-vs-baselines curve: error against the tokens actually read."""
import json, sys
for r in [json.loads(l) for l in open(sys.argv[1])]:
    M = r["methods"]; bs = list(M["oracle"].keys())
    print(f"\n### {r['tag']} | {r['task']} {r['datalen']} | M={r['M']} leaf={r['leaf']} "
          f"eta_q={r['eta_q']} pq={r['pq_mode']}")
    print("relerr cua attention output (median) -- thap hon = tot hon")
    print(f"{'budget':<9}" + "".join(f"{m:>12}" for m in M))
    print("-" * (9 + 12 * len(M)))
    for b in bs:
        print(f"{b:<9}" + "".join(f"{M[m][b]['relerr_median']:12.4f}" for m in M))
    print(f"\ntokens THUC su phai doc (cache dung chung cua GQA group)")
    print(f"{'budget':<9}" + "".join(f"{m:>12}" for m in M))
    for b in bs:
        print(f"{b:<9}" + "".join(f"{M[m][b]['tokens']:12.0f}" for m in M))
