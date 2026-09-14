#!/usr/bin/env python3
"""Compare query-free block features against oracle centroid-refinement need.

Real decode queries are used only to construct offline labels: a block needs
refinement on a step when exact block log-sum-exp is in the router top-k but
the one-mean approximation is not.  Every candidate predictor is computed
from the eight post-RoPE keys inside that block alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from scipy.stats import rankdata, spearmanr


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))

from models.centroid_router_cache import fit_self_lse_two_centroids  # noqa: E402


def parse_case(spec: str) -> tuple[Path, int, int, int]:
    path, layer, head, target = spec.rsplit(",", 3)
    return Path(path), int(layer), int(head), int(target)


def top_mask(score: torch.Tensor, k: int) -> torch.Tensor:
    """Return [steps, blocks] top-k after ShadowKV's GQA max merge."""
    merged = torch.softmax(score, dim=-1).amax(0)
    indices = merged.topk(min(k, merged.shape[-1]), dim=-1).indices
    mask = torch.zeros_like(merged, dtype=torch.bool)
    return mask.scatter_(1, indices, True)


def descending_rank(values: torch.Tensor, index: int) -> int:
    return int((values > values[index]).sum()) + 1


def auc_score(values: torch.Tensor, positive: torch.Tensor) -> float:
    labels = positive.cpu().numpy().astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(values.cpu().numpy(), method="average")
    return float(
        (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def safe_spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    value = spearmanr(left.cpu().numpy(), right.cpu().numpy()).statistic
    return float(value) if math.isfinite(value) else float("nan")


@torch.inference_mode()
def analyze_case(
    probe: Path,
    layer: int,
    head: int,
    target_block: int,
    device: str,
    top_k: int,
) -> tuple[list[dict], list[dict], dict]:
    dump = torch.load(probe, map_location="cpu", weights_only=False)
    location = (layer, head)
    blocks = dump["blocks"][location].float().to(device)
    block_ids = dump["block_ids"][location].long().to(device)
    raw_query = dump["queries"][location].float().to(device)
    query = raw_query / math.sqrt(raw_query.shape[-1])
    groups, steps, dim = query.shape
    count, block_size, _ = blocks.shape

    exact = torch.logsumexp(
        torch.einsum("gtd,nsd->gtns", query, blocks), dim=-1
    )
    mean = blocks.mean(1)
    lower_one = (
        torch.einsum("gtd,nd->gtn", query, mean) + math.log(block_size)
    )
    centers, counts, risk_one, risk_two, self_gain = (
        fit_self_lse_two_centroids(blocks, temperatures=(1.0,))
    )
    lower_two = torch.logsumexp(
        torch.einsum("gtd,nrd->gtnr", query, centers)
        + counts.float().log()[None, None],
        dim=-1,
    )

    exact_top = top_mask(exact, top_k)
    one_top = top_mask(lower_one, top_k)
    two_top = top_mask(lower_two, top_k)
    missed = exact_top & ~one_top
    repaired = missed & two_top
    actual_gain = (lower_two - lower_one).mean((0, 1))
    oracle_gap_one = (exact - lower_one).clamp_min(0)
    oracle_gap_two = (exact - lower_two).clamp_min(0)

    norms = blocks.norm(dim=-1)
    centered = blocks - mean[:, None]
    centered_norm = centered.norm(dim=-1)
    unit = torch.nn.functional.normalize(blocks, dim=-1, eps=1e-12)
    cosine = torch.einsum("nid,njd->nij", unit, unit)
    offdiag = ~torch.eye(block_size, device=blocks.device, dtype=torch.bool)
    pair_cosine = cosine[:, offdiag]
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    energy_probability = energy / energy.sum(-1, keepdim=True).clamp_min(1e-12)
    effective_rank = torch.exp(
        -(energy_probability * energy_probability.clamp_min(1e-12).log()).sum(-1)
    )

    # The response of the block to its own normalized keys.  High peakiness
    # means at least one internal direction makes the mean especially unsafe.
    self_logits = torch.einsum("njd,nid->nji", unit, blocks)
    self_probability = torch.softmax(self_logits, dim=-1)
    self_entropy = -(
        self_probability * self_probability.clamp_min(1e-12).log()
    ).sum(-1)

    features = {
        "self_lse_gain": self_gain,
        "self_lse_risk_one": risk_one,
        "self_lse_risk_two": risk_two,
        "key_norm_mean": norms.mean(-1),
        "key_norm_max": norms.amax(-1),
        "key_norm_range": norms.amax(-1) - norms.amin(-1),
        "key_norm_cv": norms.std(-1) / norms.mean(-1).clamp_min(1e-12),
        "center_radius": centered_norm.amax(-1),
        "center_rms": centered.square().sum(-1).mean(-1).sqrt(),
        "cosine_spread": 1.0 - pair_cosine.amin(-1),
        "cosine_std": pair_cosine.std(-1),
        "spectral_top1": energy_probability[:, 0],
        "spectral_top2": energy_probability[:, :2].sum(-1),
        "spectral_effective_rank": effective_rank,
        "self_peak_mean": self_probability.amax(-1).mean(-1),
        "self_peak_max": self_probability.amax(-1).amax(-1),
        "self_entropy_deficit": math.log(block_size) - self_entropy.mean(-1),
    }
    targets = {
        "actual_lse_gain": actual_gain,
        "oracle_gap_one_mean": oracle_gap_one.mean((0, 1)),
        "oracle_gap_one_max": oracle_gap_one.amax((0, 1)),
        "mean_miss_rate": missed.float().mean(0),
        "two_center_repair_rate": repaired.float().mean(0),
        "exact_top_rate": exact_top.float().mean(0),
    }
    positive = targets["mean_miss_rate"] > 0
    target_found = (block_ids == target_block).nonzero(as_tuple=False)
    if not target_found.numel():
        raise KeyError(f"target block {target_block} is absent from {location}")
    target_index = int(target_found[0, 0])
    selected = max(1, math.ceil(0.25 * count))

    feature_rows = []
    for name, values in features.items():
        chosen = values.topk(selected).indices
        recall = (
            float(positive[chosen].sum() / positive.sum())
            if positive.any()
            else float("nan")
        )
        feature_rows.append(
            {
                "sample": dump["sample_index"],
                "layer": layer,
                "head": head,
                "feature": name,
                "spearman_actual_lse_gain": safe_spearman(values, actual_gain),
                "spearman_mean_miss_rate": safe_spearman(
                    values, targets["mean_miss_rate"]
                ),
                "auc_any_mean_miss": auc_score(values, positive),
                "top25_recall_any_mean_miss": recall,
                "target_rank": descending_rank(values, target_index),
                "target_percentile": descending_rank(values, target_index) / count,
            }
        )

    block_rows = []
    cpu_features = {name: value.cpu().tolist() for name, value in features.items()}
    cpu_targets = {name: value.cpu().tolist() for name, value in targets.items()}
    for index, block_id in enumerate(block_ids.tolist()):
        block_rows.append(
            {
                "sample": dump["sample_index"],
                "layer": layer,
                "head": head,
                "block": block_id,
                "is_causal_target": block_id == target_block,
                **{name: values[index] for name, values in cpu_features.items()},
                **{name: values[index] for name, values in cpu_targets.items()},
            }
        )

    case_summary = {
        "probe": str(probe),
        "sample": dump["sample_index"],
        "layer": layer,
        "head": head,
        "target_block": target_block,
        "candidate_blocks": count,
        "query_groups": groups,
        "decode_steps": steps,
        "blocks_missed_at_least_once_by_mean": int(positive.sum()),
        "target_mean_miss_rate": float(targets["mean_miss_rate"][target_index]),
        "target_two_center_repair_rate": float(
            targets["two_center_repair_rate"][target_index]
        ),
        "target_actual_lse_gain": float(actual_gain[target_index]),
    }
    return block_rows, feature_rows, case_summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--case",
        action="append",
        required=True,
        help="PROBE_PATH,LAYER,HEAD,CAUSAL_BLOCK; repeat for each location",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    block_rows: list[dict] = []
    feature_rows: list[dict] = []
    summaries: list[dict] = []
    for spec in args.case:
        rows, features, summary = analyze_case(
            *parse_case(spec), device=args.device, top_k=args.top_k
        )
        block_rows.extend(rows)
        feature_rows.extend(features)
        summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_block.csv", block_rows)
    write_csv(args.output_dir / "feature_diagnostics.csv", feature_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"top_k": args.top_k, "cases": summaries}, indent=2)
    )
    print(json.dumps(summaries, indent=2))
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
