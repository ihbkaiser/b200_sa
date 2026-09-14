#!/usr/bin/env python
"""Render the lab2 sweep: depth against the attention-output error it buys."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
sel = sys.argv[2] if len(sys.argv) > 2 else None
for r in rows:
    if sel and r.get("tag") != sel:
        continue
    covs = list(r["oracle"].keys())
    print(f"\n### {r['tag']}  M={r['M']} leaf={r['leaf']} pq_subdim={r['pq_subdim']} "
          f"| {r['task']} {r['datalen']} | Quest meta = {r['quest_meta_frac']*100:.2f}% cua KV")
    print(f"{'envelope':<14}{'meta%':>7}{'viol%':>7} | "
          + "".join(f"{('depth@'+c):>12}{'relerr':>9}" for c in covs))
    print("-" * (28 + 21 * len(covs)))
    for n, v in r["variants"].items():
        line = f"{n:<14}{v['meta_frac_of_kv']*100:6.2f}%{v['violation_pct']:6.2f}% | "
        for c in covs:
            line += f"{v['cov'][c]['depth_pct']:11.1f}%{v['cov'][c]['relerr_median']:9.4f}"
        print(line)
    line = f"{'ORACLE':<14}{'-':>7}{'-':>7} | "
    for c in covs:
        line += f"{r['oracle'][c]['depth_pct']:11.1f}%{r['oracle'][c]['relerr_median']:9.4f}"
    print(line)
