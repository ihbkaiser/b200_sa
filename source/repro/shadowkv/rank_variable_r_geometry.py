#!/usr/bin/env python3
"""Rank exact key-only variable-r gains against realized decode-LSE gains."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))

from models.centroid_router_cache import fit_exact_key_only_r_centroids  # noqa: E402


def parse_locations(spec: str) -> set[tuple[int, int]]:
    return {
        tuple(map(int, item.split(":")))
        for item in spec.split(",")
        if item
    }


def descending_rank(values: torch.Tensor, index: int) -> int:
    return int((values > values[index]).sum().item()) + 1


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--locations", required=True)
    ap.add_argument("--target-block", type=int, required=True)
    ap.add_argument(
        "--objectives", default="minimax,scatter,cosine"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-blocks", type=int, default=32)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    dump = torch.load(args.probe, map_location="cpu", weights_only=False)
    rows: list[dict] = []
    target_rows: list[dict] = []
    for layer, head in sorted(parse_locations(args.locations)):
        blocks = dump["blocks"][(layer, head)].float().to(args.device)
        ids = dump["block_ids"][(layer, head)].long().to(args.device)
        raw_query = dump["queries"][(layer, head)].float().to(args.device)
        query = (raw_query / math.sqrt(raw_query.shape[-1])).reshape(
            -1, raw_query.shape[-1]
        )
        target_found = (ids == args.target_block).nonzero(as_tuple=False)
        if not target_found.numel():
            raise KeyError(args.target_block)
        target = int(target_found[0, 0])
        exact = torch.logsumexp(
            torch.einsum("qd,nsd->qns", query, blocks), dim=-1
        )

        value_ids = set(dump["block_sets"]["value"])
        ids_list = ids.tolist()
        for objective in args.objectives.split(","):
            by_r = {}
            for r in range(1, blocks.shape[-2] + 1):
                centers, counts, key_distortion = fit_exact_key_only_r_centroids(
                    blocks,
                    n_centroids=r,
                    objective=objective,
                    batch_blocks=args.batch_blocks,
                )
                lower = torch.logsumexp(
                    torch.einsum("qd,nrd->qnr", query, centers)
                    + counts.float().log()[None],
                    dim=-1,
                )
                gap = (exact - lower).clamp_min(0)
                by_r[r] = {
                    "key": key_distortion,
                    "gap_mean": gap.mean(0),
                    "gap_p95": torch.quantile(gap, 0.95, dim=0),
                    "gap_max": gap.amax(0),
                }

            key_one = by_r[1]["key"]
            gap_one = by_r[1]["gap_mean"]
            previous_key = key_one
            previous_gap = gap_one
            for r in range(1, blocks.shape[-2] + 1):
                item = by_r[r]
                key_gain = key_one - item["key"]
                actual_gain = gap_one - item["gap_mean"]
                marginal_key = previous_key - item["key"] if r > 1 else key_gain
                marginal_actual = (
                    previous_gap - item["gap_mean"] if r > 1 else actual_gain
                )
                if r == 1:
                    correlation = float("nan")
                else:
                    correlation = float(
                        spearmanr(
                            key_gain.cpu().numpy(), actual_gain.cpu().numpy()
                        ).statistic
                    )
                target_row = {
                    "sample": dump["sample_index"],
                    "layer": layer,
                    "head": head,
                    "target_block": args.target_block,
                    "objective": objective,
                    "r": r,
                    "target_key_distortion": float(item["key"][target]),
                    "target_lse_gap_mean": float(item["gap_mean"][target]),
                    "target_lse_gap_p95": float(item["gap_p95"][target]),
                    "target_lse_gap_max": float(item["gap_max"][target]),
                    "target_cumulative_key_gain_rank": descending_rank(key_gain, target),
                    "target_marginal_key_gain_rank": descending_rank(marginal_key, target),
                    "target_cumulative_actual_gain_rank": descending_rank(actual_gain, target),
                    "target_marginal_actual_gain_rank": descending_rank(
                        marginal_actual, target
                    ),
                    "spearman_key_gain_vs_lse_gain_all_blocks": correlation,
                }
                target_rows.append(target_row)
                cpu_columns = {
                    "key_distortion": item["key"].cpu().tolist(),
                    "key_gain_from_r1": key_gain.cpu().tolist(),
                    "marginal_key_gain": marginal_key.cpu().tolist(),
                    "lse_gap_mean": item["gap_mean"].cpu().tolist(),
                    "lse_gap_p95": item["gap_p95"].cpu().tolist(),
                    "lse_gap_max": item["gap_max"].cpu().tolist(),
                    "actual_lse_gain_from_r1": actual_gain.cpu().tolist(),
                    "marginal_actual_lse_gain": marginal_actual.cpu().tolist(),
                }
                for compact, block_id in enumerate(ids_list):
                    rows.append(
                        {
                            "sample": dump["sample_index"],
                            "layer": layer,
                            "head": head,
                            "block": block_id,
                            "is_target": block_id == args.target_block,
                            "is_value_block": block_id in value_ids,
                            "objective": objective,
                            "r": r,
                            **{
                                name: values[compact]
                                for name, values in cpu_columns.items()
                            },
                        }
                    )
                previous_key = item["key"]
                previous_gap = item["gap_mean"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("all_blocks.csv", rows), ("target_ranks.csv", target_rows)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = {
        "probe": str(args.probe),
        "target_block": args.target_block,
        "target_rows": target_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(target_rows, indent=2))
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
