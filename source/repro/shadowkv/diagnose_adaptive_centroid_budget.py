#!/usr/bin/env python3
"""Diagnose non-monotone accuracy as the adaptive centroid budget grows.

The probe runs one dense reference trajectory, rebuilds ShadowKV's candidate
blocks exactly, and evaluates the adaptive router at several centroid budgets
on the *same* post-RoPE keys and dense queries.  It therefore separates
autoregressive trajectory changes from three router questions:

1. Are per-block component counts monotone and is the global budget exact?
2. Does a larger representation improve agreement with exact block LSE?
3. Are answer-bearing blocks promoted or displaced in the top-k router?
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
    allocate_concave_marginal_counts,
    evaluate_self_lse_cvar_path,
    fit_agglomerative_self_lse_paths,
    pack_adaptive_centroids,
)
from repro.certified_sparse.probe_hard_multikey_block8 import (  # noqa: E402
    capture_dense,
    find_text_spans,
    flatten_answer,
    load_ruler_row,
)


def router_score(logits: torch.Tensor) -> torch.Tensor:
    """ShadowKV's GQA-reduced router score: [G,T,N] -> [T,N]."""
    return torch.softmax(logits.float(), dim=-1).amax(dim=0)


def top_indices(score: torch.Tensor, count: int) -> torch.Tensor:
    return score.topk(min(count, score.shape[-1]), dim=-1).indices


def set_overlap(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left[..., None] == right[:, None, :]).any(-1).float().mean(-1)


def ranks(score: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    selected = score[:, indices]
    return (score[:, None, :] > selected[..., None]).sum(-1) + 1


def segmented_lse(
    query: torch.Tensor,
    centers: torch.Tensor,
    populations: torch.Tensor,
    alpha: torch.Tensor,
    block_ids: torch.Tensor,
    n_blocks: int,
    corrected: bool,
) -> torch.Tensor:
    """Return [G,T,N] mixture LSE for one KV head."""
    component = torch.einsum("gtd,md->gtm", query, centers.float())
    component = component + populations.float().log()[None, None]
    if corrected:
        component = component + query.square().sum(-1)[..., None] * alpha.float()[
            None, None
        ]
    index = block_ids.long()[None, None].expand_as(component)
    shape = (*component.shape[:-1], n_blocks)
    maximum = torch.full(
        shape, -torch.inf, device=component.device, dtype=component.dtype
    )
    maximum.scatter_reduce_(
        -1, index, component, reduce="amax", include_self=True
    )
    centered = component - maximum.gather(-1, index)
    total = torch.zeros_like(maximum)
    total.scatter_add_(-1, index, centered.exp())
    return maximum + total.log()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--fractions", default="0.25,0.5,1.0")
    ap.add_argument("--max-decode-steps", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--dynamic-budget", type=int, default=512)
    ap.add_argument("--outlier-blocks", type=int, default=48)
    ap.add_argument("--local-blocks", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    fractions = tuple(float(item) for item in args.fractions.split(","))
    if tuple(sorted(fractions)) != fractions:
        raise ValueError("fractions must be increasing")
    if args.block_size != 8:
        raise ValueError("the exact partition solver currently requires block size 8")

    row = load_ruler_row(args.dataset, args.sample_index)
    from transformers import AutoTokenizer

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
    # QA answers need not occur verbatim in the prompt.  Evidence diagnostics
    # are optional; router overlap and retained mass remain well-defined.
    evidence_blocks = sorted(
        {position // args.block_size for position in evidence_positions}
    )

    pages = len(tokens) // args.block_size
    routed_pages = pages - args.local_blocks
    usable = pages * args.block_size
    groups = llm.num_heads // llm.num_key_value_heads
    query_all = queries.reshape(
        llm.num_layers,
        queries.shape[1],
        llm.num_key_value_heads,
        groups,
        llm.head_dim,
    ).float() / math.sqrt(llm.head_dim)
    select_count = args.dynamic_budget // args.block_size

    fields = (
        "overlap",
        "mass",
        "raw_abs_error",
        "corrected_abs_error",
        "raw_signed_error",
        "corrected_signed_error",
        "raw_over_rate",
        "corrected_over_rate",
    )
    aggregate = {
        fraction: {field: [] for field in fields} for fraction in fractions
    }
    transitions = {
        (left, right): {
            "count_violations": 0,
            "count_cells": 0,
            "raw_decrease": [],
            "corrected_decrease": [],
            "raw_delta": [],
            "corrected_delta": [],
        }
        for left, right in zip(fractions, fractions[1:])
    }
    evidence_rows: list[dict] = []
    selection_step_rows: list[dict] = []
    evidence_step_rows: list[dict] = []

    for layer in range(llm.num_layers):
        key_layer = llm.kv_cache.k_cache[layer][0, :, :usable].float()
        for head in range(llm.num_key_value_heads):
            chunks = key_layer[head].reshape(
                pages, args.block_size, llm.head_dim
            )
            routed = chunks[:routed_pages]
            block_mean = routed.mean(1)
            cosine = F.cosine_similarity(
                block_mean[:, None], routed, dim=-1
            )
            fixed = cosine.amin(-1).topk(
                min(args.outlier_blocks, routed_pages), largest=False
            ).indices
            candidate_mask = torch.ones(
                routed_pages, device="cuda", dtype=torch.bool
            )
            candidate_mask[fixed] = False
            candidate_ids = candidate_mask.nonzero().squeeze(-1)
            compact = routed[candidate_ids]
            n_blocks = compact.shape[0]

            query = (
                query_all[layer, :, head]
                .transpose(0, 1)
                .contiguous()
                .to("cuda")
            )
            # query: [G,T,D]
            exact = torch.logsumexp(
                torch.einsum("gtd,nsd->gtns", query, compact), dim=-1
            )
            exact_score = router_score(exact)
            exact_top = top_indices(exact_score, select_count)
            exact_probability = torch.softmax(exact, dim=-1)

            _, assignment_path = fit_agglomerative_self_lse_paths(
                compact[None, None],
                temperatures=(1.0,),
                cost_mode="max_gap",
            )
            risk_path = evaluate_self_lse_cvar_path(
                compact[None, None],
                assignment_path,
                temperatures=(1.0,),
                tail_fraction=0.25,
            )

            evidence_compact = []
            for block in evidence_blocks:
                location = (candidate_ids == block).nonzero(as_tuple=False)
                if location.numel():
                    evidence_compact.append(int(location[0, 0]))
            evidence_index = torch.tensor(
                evidence_compact, device="cuda", dtype=torch.long
            )

            previous = None
            for fraction in fractions:
                counts = allocate_concave_marginal_counts(
                    risk_path, extra_fraction=fraction
                )
                centers, population, alpha, component_block = (
                    pack_adaptive_centroids(
                        compact[None, None], assignment_path, counts
                    )
                )
                count = counts[0, 0].long()
                raw = segmented_lse(
                    query,
                    centers[0, 0],
                    population[0, 0],
                    alpha[0, 0],
                    component_block[0, 0],
                    n_blocks,
                    corrected=False,
                )
                corrected = segmented_lse(
                    query,
                    centers[0, 0],
                    population[0, 0],
                    alpha[0, 0],
                    component_block[0, 0],
                    n_blocks,
                    corrected=True,
                )
                score = router_score(corrected)
                chosen = top_indices(score, select_count)
                raw_score = router_score(raw)
                raw_chosen = top_indices(raw_score, select_count)
                state = aggregate[fraction]
                state["overlap"].extend(
                    set_overlap(chosen, exact_top).cpu().tolist()
                )
                gather = chosen[None].expand(groups, -1, -1)
                state["mass"].extend(
                    exact_probability.gather(-1, gather)
                    .sum(-1)
                    .cpu()
                    .flatten()
                    .tolist()
                )
                corrected_mass = (
                    exact_probability.gather(
                        -1, chosen[None].expand(groups, -1, -1)
                    )
                    .sum(-1)
                    .mean(0)
                )
                raw_mass = (
                    exact_probability.gather(
                        -1, raw_chosen[None].expand(groups, -1, -1)
                    )
                    .sum(-1)
                    .mean(0)
                )
                corrected_overlap = set_overlap(chosen, exact_top)
                raw_overlap = set_overlap(raw_chosen, exact_top)
                for step in range(query.shape[1]):
                    selection_step_rows.append(
                        {
                            "sample": args.sample_index,
                            "layer": layer,
                            "kv_head": head,
                            "step": step,
                            "fraction": fraction,
                            "mean_r": 1.0 + fraction,
                            "corrected_overlap": float(
                                corrected_overlap[step]
                            ),
                            "raw_overlap": float(raw_overlap[step]),
                            "corrected_exact_mass": float(
                                corrected_mass[step]
                            ),
                            "raw_exact_mass": float(raw_mass[step]),
                        }
                    )
                for name, proxy in (("raw", raw), ("corrected", corrected)):
                    error = proxy - exact
                    state[f"{name}_abs_error"].append(float(error.abs().mean()))
                    state[f"{name}_signed_error"].append(float(error.mean()))
                    state[f"{name}_over_rate"].append(
                        float((error > 1.0e-5).float().mean())
                    )

                if evidence_index.numel():
                    exact_rank = ranks(exact_score, evidence_index)
                    proxy_rank = ranks(score, evidence_index)
                    raw_rank = ranks(raw_score, evidence_index)
                    selected_evidence = (
                        chosen[..., None] == evidence_index[None, None]
                    ).any(1)
                    raw_selected_evidence = (
                        raw_chosen[..., None] == evidence_index[None, None]
                    ).any(1)
                    for offset, compact_index in enumerate(evidence_compact):
                        evidence_rows.append(
                            {
                                "sample": args.sample_index,
                                "layer": layer,
                                "kv_head": head,
                                "block": int(candidate_ids[compact_index]),
                                "fraction": fraction,
                                "mean_r": 1.0 + fraction,
                                "components": int(count[compact_index]),
                                "exact_rank_mean": float(
                                    exact_rank[:, offset].float().mean()
                                ),
                                "proxy_rank_mean": float(
                                    proxy_rank[:, offset].float().mean()
                                ),
                                "exact_top64_rate": float(
                                    (exact_rank[:, offset] <= select_count)
                                    .float()
                                    .mean()
                                ),
                                "proxy_top64_rate": float(
                                    selected_evidence[:, offset].float().mean()
                                ),
                                "raw_error_mean": float(
                                    (raw[..., compact_index] - exact[..., compact_index]).mean()
                                ),
                                "corrected_error_mean": float(
                                    (
                                        corrected[..., compact_index]
                                        - exact[..., compact_index]
                                    ).mean()
                                ),
                            }
                        )
                        component_mask = (
                            component_block[0, 0].long() == compact_index
                        )
                        block_alpha = alpha[0, 0][component_mask].float()
                        for step in range(query.shape[1]):
                            evidence_step_rows.append(
                                {
                                    "sample": args.sample_index,
                                    "layer": layer,
                                    "kv_head": head,
                                    "step": step,
                                    "block": int(candidate_ids[compact_index]),
                                    "fraction": fraction,
                                    "mean_r": 1.0 + fraction,
                                    "components": int(count[compact_index]),
                                    "alpha_mean": float(block_alpha.mean()),
                                    "exact_rank": int(exact_rank[step, offset]),
                                    "raw_rank": int(raw_rank[step, offset]),
                                    "corrected_rank": int(
                                        proxy_rank[step, offset]
                                    ),
                                    "exact_selected": bool(
                                        exact_rank[step, offset]
                                        <= select_count
                                    ),
                                    "raw_selected": bool(
                                        raw_selected_evidence[step, offset]
                                    ),
                                    "corrected_selected": bool(
                                        selected_evidence[step, offset]
                                    ),
                                    "exact_router_score": float(
                                        exact_score[step, compact_index]
                                    ),
                                    "raw_router_score": float(
                                        raw_score[step, compact_index]
                                    ),
                                    "corrected_router_score": float(
                                        score[step, compact_index]
                                    ),
                                }
                            )

                if previous is not None:
                    left, left_count, left_raw, left_corrected = previous
                    transition = transitions[(left, fraction)]
                    transition["count_violations"] += int(
                        (count < left_count).sum()
                    )
                    transition["count_cells"] += count.numel()
                    transition["raw_decrease"].append(
                        float((raw < left_raw - 1.0e-5).float().mean())
                    )
                    transition["corrected_decrease"].append(
                        float(
                            (corrected < left_corrected - 1.0e-5)
                            .float()
                            .mean()
                        )
                    )
                    transition["raw_delta"].append(float((raw - left_raw).mean()))
                    transition["corrected_delta"].append(
                        float((corrected - left_corrected).mean())
                    )
                previous = (fraction, count, raw, corrected)
        print(f"layer {layer + 1}/{llm.num_layers}", flush=True)

    summary_rows = []
    for fraction in fractions:
        state = aggregate[fraction]
        summary_rows.append(
            {
                "fraction": fraction,
                "mean_r": 1.0 + fraction,
                **{field: mean(state[field]) for field in fields},
            }
        )
    transition_rows = []
    for (left, right), state in transitions.items():
        transition_rows.append(
            {
                "from_fraction": left,
                "to_fraction": right,
                "count_monotonicity_violations": state["count_violations"],
                "count_cells": state["count_cells"],
                "raw_score_decrease_rate": mean(state["raw_decrease"]),
                "corrected_score_decrease_rate": mean(
                    state["corrected_decrease"]
                ),
                "raw_score_mean_delta": mean(state["raw_delta"]),
                "corrected_score_mean_delta": mean(state["corrected_delta"]),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.output / "router_summary.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(
        args.output / "budget_transitions.csv", index=False
    )
    pd.DataFrame(evidence_rows).to_csv(
        args.output / "evidence_blocks.csv", index=False
    )
    pd.DataFrame(selection_step_rows).to_csv(
        args.output / "selection_steps.csv", index=False
    )
    pd.DataFrame(evidence_step_rows).to_csv(
        args.output / "evidence_steps.csv", index=False
    )
    payload = {
        "model": args.model,
        "dataset": str(args.dataset),
        "sample_index": args.sample_index,
        "prompt_tokens": len(tokens),
        "dense_decode_steps": int(queries.shape[1]),
        "answer": answer,
        "dense_generated": tokenizer.decode(generated, skip_special_tokens=True),
        "evidence_blocks": evidence_blocks,
        "router_summary": summary_rows,
        "budget_transitions": transition_rows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print("\nRouter summary")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nBudget transitions")
    print(pd.DataFrame(transition_rows).to_string(index=False))


if __name__ == "__main__":
    main()
