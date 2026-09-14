#!/usr/bin/env python3
"""Diagnose variable-rate centroid-LSE routing on one RULER example.

The audit uses the dense model's real decode queries and prompt keys.  It asks
whether the static radius gain used by the 1/2-centroid allocator predicts the
actual Jensen gap, and whether mixing blocks at different approximation
fidelities biases the dynamic top-block ranking.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))
sys.path.insert(0, str(REPO))

from models.centroid_router_cache import (  # noqa: E402
    fit_block_centroids,
    fit_key_only_two_centroids,
    fit_minimax_two_centroids,
)
from repro.certified_sparse.probe_hard_multikey_block8 import (  # noqa: E402
    capture_dense,
    find_text_spans,
    flatten_answer,
    load_ruler_row,
)


def rankdata(x: torch.Tensor) -> torch.Tensor:
    """Deterministic ordinal ranks; sufficient for a no-tie continuous probe."""
    order = x.argsort()
    ranks = torch.empty_like(x, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    rx, ry = rankdata(x.flatten()), rankdata(y.flatten())
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = rx.norm() * ry.norm()
    return float((rx @ ry / denom).item()) if denom > 0 else float("nan")


def selected(proxy: torch.Tensor, count: int) -> torch.Tensor:
    """Reproduce ShadowKV's softmax-per-query-head then GQA-max router."""
    router = torch.softmax(proxy.float(), dim=-1).amax(dim=1)
    return router.topk(min(count, proxy.shape[-1]), dim=-1).indices


def set_overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[..., None] == b[:, None, :]).any(-1).float().mean(-1)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sample-index", type=int, default=5)
    ap.add_argument("--max-decode-steps", type=int, default=40)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--dynamic-budget", type=int, default=512)
    ap.add_argument("--outlier-blocks", type=int, default=48)
    ap.add_argument("--local-blocks", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.block_size != 8:
        raise ValueError("the exact 127-partition solver requires block size 8")

    from transformers import AutoTokenizer

    row = load_ruler_row(args.dataset, args.sample_index)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = tokenizer.encode(row["input"], add_special_tokens=False)
    input_ids = torch.tensor(tokens, device="cuda:0", dtype=torch.long)[None]
    llm, queries, _, generated = capture_dense(
        args.model, input_ids, args.max_decode_steps
    )

    answer = flatten_answer(row["outputs"])
    evidence_positions = {
        position
        for start, stop in find_text_spans(tokenizer, tokens, answer)
        for position in range(start, stop)
    }
    evidence_blocks = {position // args.block_size for position in evidence_positions}

    block = args.block_size
    pages = len(tokens) // block
    routed_pages = pages - args.local_blocks
    usable = pages * block
    groups = llm.num_heads // llm.num_key_value_heads
    query_all = queries.reshape(
        llm.num_layers,
        queries.shape[1],
        llm.num_key_value_heads,
        groups,
        llm.head_dim,
    ).float() / math.sqrt(llm.head_dim)
    select_count = args.dynamic_budget // block
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)

    method_acc = {
        fraction: {
            "overlap": [],
            "mass": [],
            "evidence": [],
            "oracle_unsplit": [],
            "selected_split": [],
            "evidence_split": [],
        }
        for fraction in fractions
    }
    correlations = []
    feature_correlations = []
    calibration = []
    alternative_acc = {
        (name, fraction): {"overlap": [], "mass": []}
        for name in (
            "minimax_radius",
            "minimax_sse",
            "scatter2",
            "cosine2",
            "minimax_oracle_query_gain",
            "kmeans_sse",
        )
        for fraction in fractions
    }

    for layer in range(llm.num_layers):
        key_layer = llm.kv_cache.k_cache[layer][0, :, :usable].float()
        for head in range(llm.num_key_value_heads):
            chunks = key_layer[head].reshape(pages, block, llm.head_dim)
            routed = chunks[:routed_pages]
            means_all = routed.mean(1)
            cosine = F.cosine_similarity(means_all[:, None], routed, dim=-1)
            fixed = cosine.amin(-1).topk(
                min(args.outlier_blocks, routed_pages), largest=False
            ).indices
            candidate_mask = torch.ones(routed_pages, device="cuda", dtype=torch.bool)
            candidate_mask[fixed] = False
            candidate_idx = candidate_mask.nonzero().squeeze(-1)
            compact = routed[candidate_idx]

            centers, counts, radius_one, radius_two = fit_minimax_two_centroids(
                compact
            )
            kmeans_centers, kmeans_counts = fit_block_centroids(
                compact, n_centroids=2, method="kmeans"
            )
            scatter_centers, scatter_counts, _, _, scatter_gain = (
                fit_key_only_two_centroids(compact, objective="scatter")
            )
            cosine_centers, cosine_counts, _, _, cosine_gain = (
                fit_key_only_two_centroids(compact, objective="cosine")
            )
            gain = radius_one - radius_two
            means = compact.mean(1)
            q = query_all[layer, :, head].to("cuda")
            token_logits = torch.einsum("agd,csd->agcs", q, compact)
            exact_lse = torch.logsumexp(token_logits, dim=-1)
            mean_lse = torch.einsum("agd,cd->agc", q, means) + math.log(block)
            two_lse = torch.logsumexp(
                torch.einsum("agd,crd->agcr", q, centers)
                + counts.float().log()[None, None],
                dim=-1,
            )
            kmeans_lse = torch.logsumexp(
                torch.einsum("agd,crd->agcr", q, kmeans_centers)
                + kmeans_counts.float().log()[None, None],
                dim=-1,
            )
            scatter_lse = torch.logsumexp(
                torch.einsum("agd,crd->agcr", q, scatter_centers)
                + scatter_counts.float().log()[None, None],
                dim=-1,
            )
            cosine_lse = torch.logsumexp(
                torch.einsum("agd,crd->agcr", q, cosine_centers)
                + cosine_counts.float().log()[None, None],
                dim=-1,
            )
            actual_gap = (exact_lse - mean_lse).mean((0, 1))
            actual_improvement = (two_lse - mean_lse).mean((0, 1))
            residual_gap = (exact_lse - two_lse).mean((0, 1))
            one_sse = (compact - means[:, None]).square().sum(-1).mean(-1)
            two_sse = (
                (compact[:, :, None] - centers[:, None])
                .square().sum(-1).amin(-1).mean(-1)
            )
            kmeans_two_sse = (
                (compact[:, :, None] - kmeans_centers[:, None])
                .square().sum(-1).amin(-1).mean(-1)
            )
            static_features = {
                "radius_one": radius_one,
                "radius_gain": gain,
                "squared_radius_gain": radius_one.square() - radius_two.square(),
                "relative_radius_gain": gain / radius_one.clamp_min(1e-8),
                "mean_sse": one_sse,
                "nearest_two_sse_gain": one_sse - two_sse,
                "normalized_scatter_gain": scatter_gain,
                "spherical_cosine_gain": cosine_gain,
            }
            for feature_name, feature in static_features.items():
                feature_correlations.append(
                    {
                        "layer": layer,
                        "kv_head": head,
                        "feature": feature_name,
                        "vs_mean_jensen_gap": spearman(feature, actual_gap),
                        "vs_actual_lse_improvement": spearman(
                            feature, actual_improvement
                        ),
                        "vs_residual_jensen_gap": spearman(feature, residual_gap),
                    }
                )
            correlations.append(
                {
                    "layer": layer,
                    "kv_head": head,
                    "radius_gain_vs_mean_jensen_gap": spearman(gain, actual_gap),
                    "radius_gain_vs_actual_lse_improvement": spearman(
                        gain, actual_improvement
                    ),
                    "radius_gain_vs_residual_jensen_gap": spearman(
                        gain, residual_gap
                    ),
                }
            )

            oracle_sel = selected(exact_lse, select_count)
            all_logmass = torch.logsumexp(exact_lse, dim=-1)
            alternatives = {
                "minimax_radius": (two_lse, gain),
                "minimax_sse": (two_lse, one_sse - two_sse),
                "scatter2": (scatter_lse, scatter_gain),
                "cosine2": (cosine_lse, cosine_gain),
                "minimax_oracle_query_gain": (two_lse, actual_improvement),
                "kmeans_sse": (kmeans_lse, one_sse - kmeans_two_sse),
            }
            for name, (refined_lse, allocation_score) in alternatives.items():
                for fraction in fractions:
                    n_split = round(fraction * compact.shape[0])
                    split = torch.zeros_like(gain, dtype=torch.bool)
                    if n_split:
                        split[allocation_score.topk(n_split).indices] = True
                    proxy = torch.where(split[None, None], refined_lse, mean_lse)
                    chosen = selected(proxy, select_count)
                    state = alternative_acc[(name, fraction)]
                    state["overlap"].extend(
                        set_overlap(chosen, oracle_sel).cpu().tolist()
                    )
                    gather = chosen[:, None].expand(-1, groups, -1)
                    captured = torch.exp(
                        torch.logsumexp(exact_lse.gather(2, gather), dim=-1)
                        - all_logmass
                    )
                    state["mass"].extend(captured.cpu().flatten().tolist())
            ev_compact = torch.tensor(
                [
                    i
                    for i, original in enumerate(candidate_idx.tolist())
                    if original in evidence_blocks
                ],
                device="cuda",
                dtype=torch.long,
            )

            for fraction in fractions:
                n_split = round(fraction * compact.shape[0])
                split = torch.zeros_like(gain, dtype=torch.bool)
                if n_split:
                    split[gain.topk(n_split).indices] = True
                proxy = torch.where(split[None, None], two_lse, mean_lse)
                chosen = selected(proxy, select_count)
                state = method_acc[fraction]
                state["overlap"].extend(set_overlap(chosen, oracle_sel).cpu().tolist())
                gather = chosen[:, None].expand(-1, groups, -1)
                captured = torch.exp(
                    torch.logsumexp(exact_lse.gather(2, gather), dim=-1)
                    - all_logmass
                )
                state["mass"].extend(captured.cpu().flatten().tolist())
                state["selected_split"].extend(
                    split[chosen].float().cpu().flatten().tolist()
                )
                state["oracle_unsplit"].extend(
                    (~split[oracle_sel]).float().cpu().flatten().tolist()
                )
                if ev_compact.numel():
                    ev_hit = (chosen[..., None] == ev_compact[None, None, :]).any(1).float().mean()
                    state["evidence"].append(float(ev_hit))
                    state["evidence_split"].append(float(split[ev_compact].float().mean()))

                # A split block has a systematically tighter lower estimate.
                if 0 < n_split < compact.shape[0]:
                    err = exact_lse - proxy
                    calibration.append(
                        {
                            "fraction": fraction,
                            "layer": layer,
                            "kv_head": head,
                            "split_underestimate": float(err[..., split].mean()),
                            "unsplit_underestimate": float(err[..., ~split].mean()),
                            "split_score_lift": float(
                                (two_lse - mean_lse)[..., split].mean()
                            ),
                        }
                    )
        print(f"layer {layer + 1}/{llm.num_layers}", flush=True)

    summary_rows = []
    for fraction, state in method_acc.items():
        summary_rows.append(
            {
                "split_fraction": fraction,
                "mean_centroids": 1.0 + fraction,
                "topset_overlap_with_exact_lse": sum(state["overlap"]) / len(state["overlap"]),
                "candidate_attention_mass_capture": sum(state["mass"]) / len(state["mass"]),
                "evidence_block_recall": (
                    sum(state["evidence"]) / len(state["evidence"])
                    if state["evidence"] else float("nan")
                ),
                "exact_topset_left_unsplit": sum(state["oracle_unsplit"]) / len(state["oracle_unsplit"]),
                "selected_blocks_that_are_split": sum(state["selected_split"]) / len(state["selected_split"]),
                "candidate_evidence_blocks_that_are_split": (
                    sum(state["evidence_split"]) / len(state["evidence_split"])
                    if state["evidence_split"] else float("nan")
                ),
            }
        )

    alternative_rows = []
    for (name, fraction), state in alternative_acc.items():
        alternative_rows.append(
            {
                "partition_allocator": name,
                "split_fraction": fraction,
                "mean_centroids": 1.0 + fraction,
                "topset_overlap_with_exact_lse": sum(state["overlap"]) / len(state["overlap"]),
                "candidate_attention_mass_capture": sum(state["mass"]) / len(state["mass"]),
            }
        )

    corr_frame = pd.DataFrame(correlations)
    feature_corr_frame = pd.DataFrame(feature_correlations)
    cal_frame = pd.DataFrame(calibration)
    summary_frame = pd.DataFrame(summary_rows)
    alternative_frame = pd.DataFrame(alternative_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    corr_frame.to_csv(args.output / "per_layer_head_correlations.csv", index=False)
    feature_corr_frame.to_csv(args.output / "static_feature_correlations.csv", index=False)
    cal_frame.to_csv(args.output / "per_layer_head_calibration.csv", index=False)
    summary_frame.to_csv(args.output / "router_summary.csv", index=False)
    alternative_frame.to_csv(args.output / "allocator_comparison.csv", index=False)
    payload = {
        "model": args.model,
        "dataset": str(args.dataset),
        "sample_index": args.sample_index,
        "prompt_tokens": len(tokens),
        "decode_steps": int(queries.shape[1]),
        "answer": answer,
        "dense_generated": tokenizer.decode(generated, skip_special_tokens=True),
        "mean_correlations": corr_frame.mean(numeric_only=True).to_dict(),
        "static_feature_correlations": feature_corr_frame.groupby("feature").mean(numeric_only=True).reset_index().to_dict("records"),
        "mean_calibration": cal_frame.groupby("fraction").mean(numeric_only=True).reset_index().to_dict("records"),
        "router_summary": summary_rows,
        "allocator_comparison": alternative_rows,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nRouter summary")
    print(summary_frame.to_string(index=False))
    print("\nMean correlations")
    print(corr_frame.mean(numeric_only=True).to_string())
    print("\nStatic feature correlations")
    print(feature_corr_frame.groupby("feature").mean(numeric_only=True).to_string())
    print("\nCalibration bias")
    print(cal_frame.groupby("fraction").mean(numeric_only=True).to_string())
    print("\nAllocator comparison")
    print(alternative_frame.to_string(index=False))


if __name__ == "__main__":
    main()
