#!/usr/bin/env python
"""
Compare the raw predictions of cells that differ only in method/budget.

Agreement is reported against full attention and between budget pairs.
~100% agreement where the budget was cut hard is the signature of a method
that is not actually compressing -- a score table cannot show you that.
"""

import argparse
import json
import os
import re


def load(path):
    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            preds.extend(row.get("prediction", []))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--datalen", type=int, required=True)
    args = ap.parse_args()

    prefix = f"{args.task}_{args.datalen}_"
    cells = {}
    for fn in sorted(os.listdir(args.dir)):
        if fn.startswith(prefix) and fn.endswith(".jsonl"):
            cells[fn[len(prefix):-len(".jsonl")]] = load(os.path.join(args.dir, fn))

    if "full" not in cells:
        raise SystemExit(f"no full-attention cell in {args.dir} for {prefix}")

    ref = cells["full"]
    distinct_ref = len({p.strip() for p in ref})
    print(f"\n=== predictions vs full attention: {args.task} @ {args.datalen}, {len(ref)} samples ===")
    print(f"    dense reference has {distinct_ref} distinct answer(s) over {len(ref)} sample(s)")
    if distinct_ref <= 1 or len(ref) < 8:
        print("    NOTE: with few samples, or a task the model answers the same way every time,\n"
              "          agreement with dense is expected and proves nothing either way.\n"
              "          Use budget_audit.py for the structural check.")
    for name, preds in sorted(cells.items()):
        if name == "full":
            continue
        n = min(len(ref), len(preds))
        same = sum(1 for a, b in zip(ref[:n], preds[:n]) if a.strip() == b.strip())
        verdict = "matches dense" if n and same == n else "differs from dense"
        print(f"  {name:<28} {same}/{n} identical   {verdict}")

    # budget pairs within a method
    by_method = {}
    for name in cells:
        m = re.match(r"(quest_stream|shadowkv|shadowkv_cpu)_b(\d+)", name)
        if m:
            by_method.setdefault(m.group(1), []).append((int(m.group(2)), name))
    print("\n=== does the budget flag bite? ===")
    for method, entries in sorted(by_method.items()):
        entries.sort()
        for (b1, n1), (b2, n2) in zip(entries, entries[1:]):
            a, b = cells[n1], cells[n2]
            n = min(len(a), len(b))
            same = sum(1 for x, y in zip(a[:n], b[:n]) if x.strip() == y.strip())
            if n and same == n:
                verdict = ("no visible effect on this cell -- inconclusive; "
                           "confirm with budget_audit.py that the attended-token "
                           "count actually differs")
            else:
                verdict = "budget changes the output"
            print(f"  {method}: b{b1} vs b{b2}   {same}/{n} identical   {verdict}")
    print("\nA score table cannot tell a working method from a silent fallback.\n"
          "budget_audit.py answers the structural question (how many tokens are\n"
          "actually attended); this script answers the behavioural one.\n")


if __name__ == "__main__":
    main()
