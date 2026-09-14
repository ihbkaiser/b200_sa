#!/usr/bin/env python3
"""Measure how many within-block support modes real decode queries visit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


def parse_locations(spec: str) -> set[tuple[int, int]]:
    return {tuple(map(int, item.split(":"))) for item in spec.split(",") if item}


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--locations", required=True)
    ap.add_argument("--target-block", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    dump = torch.load(args.probe, map_location="cpu", weights_only=False)
    rows = []
    summaries = []
    value_ids = set(dump["block_sets"]["value"])
    for layer, head in sorted(parse_locations(args.locations)):
        keys = dump["blocks"][(layer, head)].float().to(args.device)
        ids = dump["block_ids"][(layer, head)].long().to(args.device)
        query = dump["queries"][(layer, head)].float().to(args.device)
        query = query / math.sqrt(query.shape[-1])
        logits = torch.einsum("gtd,nsd->gtns", query, keys)
        exact = torch.logsumexp(logits, dim=-1)
        probability = torch.softmax(exact, dim=-1)
        merged, winning_group = probability.max(0)
        top64 = merged.topk(64, dim=-1).indices
        steps = torch.arange(query.shape[1], device=args.device)
        active_counts = []
        for compact, block_id in enumerate(ids.tolist()):
            selected = (top64 == compact).any(-1)
            if not selected.any():
                continue
            chosen_logits = logits[winning_group[:, compact], steps, compact]
            offsets = chosen_logits.argmax(-1)
            sharpness = torch.softmax(chosen_logits, dim=-1).max(-1).values
            active = offsets[selected].unique()
            active_counts.append(len(active))
            rows.append(
                {
                    "sample": dump["sample_index"],
                    "layer": layer,
                    "head": head,
                    "block": block_id,
                    "kind": (
                        "causal"
                        if block_id == args.target_block
                        else "value"
                        if block_id in value_ids
                        else "other"
                    ),
                    "selected_steps": int(selected.sum()),
                    "active_offsets": "-".join(map(str, active.tolist())),
                    "active_mode_count": len(active),
                    "within_block_top1_share_mean": float(sharpness[selected].mean()),
                    "within_block_top1_share_min": float(sharpness[selected].min()),
                }
            )
        counts = torch.tensor(active_counts, dtype=torch.float32)
        causal = next(
            row
            for row in rows
            if row["sample"] == dump["sample_index"]
            and row["layer"] == layer
            and row["head"] == head
            and row["block"] == args.target_block
        )
        summaries.append(
            {
                "sample": dump["sample_index"],
                "layer": layer,
                "head": head,
                "causal_active_modes": causal["active_mode_count"],
                "causal_top1_share_mean": causal["within_block_top1_share_mean"],
                "all_selected_blocks_active_modes_median": float(counts.median()),
                "all_selected_blocks_active_modes_p90": float(
                    torch.quantile(counts, 0.9)
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summaries, indent=2)
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
