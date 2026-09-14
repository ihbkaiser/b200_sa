#!/usr/bin/env python3
"""Analyze exact variable-r centroid distortion on causal block probes.

For selected layer/KV-head locations, this script enumerates every partition
of each sampled block of eight.  It therefore separates three quantities:
(1) the globally best query-free key partition, (2) an oracle partition fitted
to the observed correct decode trajectory, and (3) the valid Hoeffding width
of the centroid-mixture Jensen lower bound.  Needle labels choose diagnostic
rows only; no label enters a deployable score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))

from models.centroid_router_cache import _restricted_growth_partitions  # noqa: E402


def parse_locations(spec: str) -> set[tuple[int, int]]:
    result = set()
    for item in spec.split(","):
        layer, head = item.split(":")
        result.add((int(layer), int(head)))
    return result


def partition_family(
    keys: torch.Tensor, queries: torch.Tensor, r: int
) -> dict[str, torch.Tensor]:
    """Return every r-partition and its exact key/query objectives."""
    block_size, dim = keys.shape
    labels = torch.tensor(
        _restricted_growth_partitions(block_size, r),
        device=keys.device,
        dtype=torch.long,
    )
    membership = torch.nn.functional.one_hot(labels, num_classes=r).float()
    counts = membership.sum(1)
    sums = torch.einsum("psr,sd->prd", membership, keys)
    centers = sums / counts[..., None]

    energy = keys.square().sum()
    scatter = (
        energy - (sums.square().sum(-1) / counts).sum(-1)
    ).clamp_min(0) / energy.clamp_min(1e-12)
    unit = torch.nn.functional.normalize(keys, dim=-1, eps=1e-12)
    unit_sums = torch.einsum("psr,sd->prd", membership, unit)
    cosine = (float(block_size) - unit_sums.norm(dim=-1).sum(-1)) / float(
        block_size
    )

    dist2 = (
        keys.square().sum(-1)[None, :, None]
        + centers.square().sum(-1)[:, None, :]
        - 2.0 * torch.einsum("sd,prd->psr", keys, centers)
    ).clamp_min(0)
    minimax = dist2.masked_fill(~membership.bool(), -torch.inf).amax((1, 2)).sqrt()

    pairwise = torch.cdist(keys, keys)
    same = torch.einsum("pir,pjr->pij", membership, membership).bool()
    diameter = pairwise[None].masked_fill(~same, -torch.inf).amax((1, 2))

    logits = torch.einsum("qd,prd->qpr", queries, centers)
    lower = torch.logsumexp(logits + counts.log()[None], dim=-1)
    exact = torch.logsumexp(torch.einsum("qd,sd->qs", queries, keys), dim=-1)
    gap = (exact[:, None] - lower).clamp_min(0)
    return {
        "labels": labels,
        "counts": counts,
        "scatter": scatter,
        "cosine": cosine,
        "minimax": minimax,
        "diameter": diameter,
        "lower": lower,
        "gap": gap,
    }


def router_rank(score: torch.Tensor, block: int) -> torch.Tensor:
    """Rank one block for [groups, steps, blocks] logits."""
    probability = torch.softmax(score, dim=-1)
    merged = probability.max(0).values
    return (merged > merged[:, block, None]).sum(-1) + 1


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--locations", required=True, help="comma-separated layer:head")
    ap.add_argument("--target-block", type=int, required=True)
    ap.add_argument("--controls", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    dump = torch.load(args.probe, map_location="cpu", weights_only=False)
    locations = parse_locations(args.locations)
    rows: list[dict] = []
    summaries: list[dict] = []
    generator = torch.Generator().manual_seed(20260903)

    for location in sorted(locations):
        if location not in dump["blocks"]:
            raise KeyError(f"location {location} absent from probe")
        blocks = dump["blocks"][location].float().to(args.device)
        block_ids = dump["block_ids"][location].long().to(args.device)
        raw_query = dump["queries"][location].float().to(args.device)
        groups, steps, dim = raw_query.shape
        query = raw_query / math.sqrt(dim)
        flat_query = query.reshape(-1, dim)
        found = (block_ids == args.target_block).nonzero(as_tuple=False)
        if not found.numel():
            raise KeyError(f"target block {args.target_block} is not a candidate")
        target_compact = int(found[0, 0])

        exact_all = torch.logsumexp(
            torch.einsum("gtd,nsd->gtns", query, blocks), dim=-1
        )
        mean_mass = torch.softmax(exact_all, dim=-1).mean((0, 1))
        excluded = set(dump["block_sets"]["value"])
        order = mean_mass.argsort(descending=True).tolist()
        top_controls = [
            index for index in order if int(block_ids[index]) not in excluded
        ][: args.controls]
        available = torch.tensor(
            [
                index
                for index, block_id in enumerate(block_ids.tolist())
                if block_id not in excluded and index not in top_controls
            ]
        )
        perm = torch.randperm(len(available), generator=generator)[: args.controls]
        random_controls = available[perm].tolist()
        value_indices = [
            int((block_ids == block).nonzero(as_tuple=False)[0, 0])
            for block in dump["block_sets"]["value"]
            if block != args.target_block and (block_ids == block).any()
        ]
        selected = []
        for kind, indices in (
            ("causal", [target_compact]),
            ("same_value", value_indices),
            ("top_mass_control", top_controls),
            ("random_control", random_controls),
        ):
            for index in indices:
                item = (kind, index)
                if item not in selected:
                    selected.append(item)

        exact_target_rank = router_rank(exact_all, target_compact)
        location_summary = {
            "layer": location[0],
            "head": location[1],
            "steps": steps,
            "groups": groups,
            "candidate_blocks": len(blocks),
            "target_block": args.target_block,
            "target_exact_rank_median": float(exact_target_rank.float().median()),
            "target_exact_top64_fraction": float((exact_target_rank <= 64).float().mean()),
        }

        for kind, compact in selected:
            key = blocks[compact]
            exact = exact_all[:, :, compact].reshape(-1)
            for r in range(1, key.shape[0] + 1):
                family = partition_family(key, flat_query, r)
                oracle_mean = family["gap"].mean(0).argmin()
                oracle_max = family["gap"].amax(0).argmin()
                choices = {
                    "minimax": family["minimax"].argmin(),
                    "scatter": family["scatter"].argmin(),
                    "cosine": family["cosine"].argmin(),
                    "oracle_mean_gap": oracle_mean,
                    "oracle_max_gap": oracle_max,
                }
                for choice_name, choice_tensor in choices.items():
                    choice = int(choice_tensor)
                    gap = family["gap"][:, choice]
                    lower = family["lower"][:, choice].reshape(groups, steps)
                    modified = exact_all.clone()
                    modified[:, :, compact] = lower
                    approximate_rank = router_rank(modified, compact)
                    qnorm2 = flat_query.square().sum(-1)
                    hoeffding = qnorm2 * family["diameter"][choice].square() / 8.0
                    rows.append(
                        {
                            "sample": dump["sample_index"],
                            "layer": location[0],
                            "head": location[1],
                            "block": int(block_ids[compact]),
                            "block_kind": kind,
                            "r": r,
                            "partition_objective": choice_name,
                            "partition_counts": "-".join(
                                map(str, family["counts"][choice].long().tolist())
                            ),
                            "key_minimax_radius": float(family["minimax"][choice]),
                            "key_scatter": float(family["scatter"][choice]),
                            "key_cosine_scatter": float(family["cosine"][choice]),
                            "cluster_diameter": float(family["diameter"][choice]),
                            "lse_gap_mean": float(gap.mean()),
                            "lse_gap_p95": float(torch.quantile(gap, 0.95)),
                            "lse_gap_max": float(gap.max()),
                            "hoeffding_width_mean": float(hoeffding.mean()),
                            "exact_top64_fraction": float(
                                (router_rank(exact_all, compact) <= 64).float().mean()
                            ),
                            "approx_top64_fraction": float(
                                (approximate_rank <= 64).float().mean()
                            ),
                            "rank_agreement": float(
                                (
                                    approximate_rank
                                    == router_rank(exact_all, compact)
                                ).float().mean()
                            ),
                        }
                    )
        summaries.append(location_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "variable_r_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"probe": str(args.probe), "locations": summaries}, indent=2)
    )
    print(json.dumps(summaries, indent=2))
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
