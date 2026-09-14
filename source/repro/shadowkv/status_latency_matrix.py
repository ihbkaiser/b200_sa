#!/usr/bin/env python3
"""Print completed rows from the isolated latency matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "adaptive_centroid_lse_streaming_prefix4_querymean": "Ours",
    "quest_streaming": "Quest",
    "shadowkv_cpu": "ShadowKV",
    "pariskv_author_common": "ParisKV",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--expected", type=int, default=12)
    args = parser.parse_args()
    records = []
    if args.results.is_file():
        for line in args.results.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"completed={len(records)}/{args.expected}")
    print(
        "Length  Method     Prefill-p50(s)  Decode-p50  Decode-mean  "
        "Decode-p95   Peak-alloc  Peak-reserved  Host-RSS"
    )
    for row in sorted(
        records,
        key=lambda item: (
            item["datalen"],
            tuple(LABELS).index(item["method"]),
        ),
    ):
        print(
            f"{row['datalen'] // 1024:>5}K  "
            f"{LABELS[row['method']]:<10} "
            f"{row['prefill_s']:>14.3f}  "
            f"{row['decode_ms']:>10.2f}  "
            f"{row['decode_mean_ms']:>11.2f}  "
            f"{row['decode_p95_ms']:>10.2f}  "
            f"{row['peak_gib']:>9.2f}G  "
            f"{row['peak_reserved_gib']:>12.2f}G  "
            f"{row['peak_host_rss_gib']:>7.2f}G"
        )


if __name__ == "__main__":
    main()
