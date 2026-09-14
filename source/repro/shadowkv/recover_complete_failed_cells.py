#!/usr/bin/env python
"""Recover pool markers only when a failed cell has complete audited output.

This is intentionally conservative.  It exists for orchestration failures
that happen after ``eval_acc.py`` has already atomically written all scores and
traffic.  A partial or configuration-mismatched cell remains failed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def last_json(path: Path):
    last = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    return last


def valid_complete(root: Path, key: str, expected: int) -> tuple[bool, str]:
    results = list((root / "cells").glob(f"*/{key}.jsonl"))
    if len(results) != 1:
        return False, f"result matches={len(results)}"
    result = results[0]
    stamp_path = result.with_name(result.stem + ".stamp.json")
    traffic_path = result.with_name(result.stem + ".traffic.jsonl")
    try:
        row = last_json(result)
        stamp = json.loads(stamp_path.read_text())
        traffic = last_json(traffic_path)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        return False, str(error)
    scores = row.get("correct") if row else None
    if not isinstance(scores, list) or len(scores) != expected:
        return False, f"scores={len(scores) if isinstance(scores, list) else 0}"
    if stamp.get("cell") != key or int(stamp.get("num_samples", -1)) != expected:
        return False, "stamp mismatch"

    method = stamp.get("method")
    budget = int(stamp.get("datalen", 0)) // 32
    if int(stamp.get("sparse_budget", -1)) != budget:
        return False, "budget mismatch"
    if method == "pqcache_author_common":
        ok = (
            stamp.get("pqcache_pq") == [2, 6]
            and stamp.get("pqcache_seed") == 4321
            and traffic.get("pqcache_seed") == 4321
            and traffic.get("pqcache_active_token_cap") == budget + 32
            and "_pq2x6_i10_x32_recent0.5_s4321_exactkv" in key
        )
    elif method == "magicpig_author_common":
        ok = (
            stamp.get("magicpig_k_l") == [10, 210]
            and stamp.get("magicpig_seed") == 43
            and traffic.get("magicpig_seed") == 43
            and "_k10l210_x4_l64_d0_s43" in key
        )
    elif method == "infllm_author_common":
        ok = (
            traffic.get("infllm_active_token_cap") == budget
            and "_blk128_repr4_native21" in key
        )
    else:
        return False, f"unsupported method={method}"
    return (True, "complete") if ok else (False, "configuration/traffic mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--state-name", default=".state_m1")
    parser.add_argument("--expected-samples", type=int, default=100)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    state = args.root / args.state_name
    recovered = 0
    for failed in sorted((state / "failed").glob("*")):
        valid, reason = valid_complete(args.root, failed.name, args.expected_samples)
        if not valid:
            print(f"KEEP {failed.name}: {reason}")
            continue
        print(f"{'RECOVER' if args.apply else 'WOULD_RECOVER'} {failed.name}")
        if args.apply:
            os.replace(failed, state / "done" / failed.name)
        recovered += 1
    print(f"recovered={recovered} apply={args.apply}")


if __name__ == "__main__":
    main()
