#!/usr/bin/env python3
"""Measure how causal prompt-query center fits generalize over decode steps.

The sparse generation trace supplies *evaluation queries only*.  Every fitted
variant uses post-RoPE queries already observed during prefill.  This isolates
whether failures of a final-query objective come from later query drift rather
than from its fit at the first decode step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_router_mass_full_trace import _concave_marginal_counts  # noqa: E402
from models.adaptive_centroid_streaming_cache import (  # noqa: E402
    _padded_adaptive_centroids,
)
from models.centroid_router_cache import fit_observed_query_mass_paths  # noqa: E402


def load_row(path: Path, index: int) -> dict:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(index)


def parse_locations(spec: str) -> set[tuple[int, int]]:
    return {
        tuple(map(int, item.split(":")))
        for item in spec.split(",") if item
    }


def reduce_gqa(query: torch.Tensor, kv_heads: int) -> torch.Tensor:
    """Convert [B,QH,T,D] post-RoPE queries to [B,KVH,T,D]."""
    batch, query_heads, tokens, dim = query.shape
    if query_heads % kv_heads:
        raise ValueError("query heads are incompatible with KV heads")
    return query.view(
        batch, kv_heads, query_heads // kv_heads, tokens, dim
    ).float().mean(2)


def causal_banks(window: torch.Tensor) -> dict[str, torch.Tensor]:
    """Small banks tied to the final prompt question, not uniform context."""
    output: dict[str, torch.Tensor] = {"final1": window[..., -1:, :]}
    for count in (2, 4, 8, 16):
        if window.shape[-2] >= count:
            output[f"recent{count}"] = window[..., -count:, :]

    # Select prompt queries most aligned with the final question query.  The
    # selection is per KV head and causal; it never observes decode queries.
    final = torch.nn.functional.normalize(window[..., -1, :], dim=-1)
    unit = torch.nn.functional.normalize(window, dim=-1)
    similarity = torch.einsum("bhtd,bhd->bht", unit, final)
    for count in (4, 8, 16):
        count = min(count, window.shape[-2])
        index = similarity.topk(count, dim=-1).indices
        gathered = window.gather(
            -2, index[..., None].expand(*index.shape, window.shape[-1])
        )
        output[f"similar{count}"] = gathered
    return output


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--datalen", type=int, default=131072)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--mean-components", type=float, default=1.5)
    ap.add_argument("--prompt-window", type=int, default=128)
    ap.add_argument("--locations", default="23:4,27:0,30:6")
    ap.add_argument(
        "--objectives", default="global_mass",
        help=(
            "comma-separated fit objectives: global_mass,"
            "global_mass_cvar,topk_hinge,topk_hinge_cvar"
        ),
    )
    ap.add_argument("--tail-fraction", type=float, default=0.0625)
    ap.add_argument(
        "--bank-names", default="",
        help="optional comma-separated subset of causal/oracle banks",
    )
    ap.add_argument(
        "--evidence-blocks", default="",
        help="optional comma-separated absolute block ids for recall audit",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    locations = parse_locations(args.locations)
    objectives = tuple(value for value in args.objectives.split(",") if value)
    if not objectives or any(
        value not in {
            "global_mass", "global_mass_cvar", "topk_hinge",
            "topk_hinge_cvar",
        }
        for value in objectives
    ):
        raise ValueError("invalid objective list")
    bank_names = {
        value for value in args.bank_names.split(",") if value
    }
    evidence_blocks = {
        int(value) for value in args.evidence_blocks.split(",") if value
    }

    os.environ["SHADOWKV_CENTER_PLACEMENT"] = "robust_trimmed"
    os.environ["SHADOWKV_CENTER_ALLOCATION"] = "tail_cvar"
    os.environ["SHADOWKV_CENTER_DISPERSION_CORRECTION"] = "0"

    from models import choose_model_class
    from models.tensor_op import sample_token

    row = load_row(args.dataset, args.sample_index)
    llm = choose_model_class(args.model)(
        model_name=args.model,
        batch_size=1,
        device="cuda:0",
        max_length=args.datalen + 2048,
        attn_mode="adaptive_centroid_lse_streaming_prefix4_querymean",
        dtype=torch.bfloat16,
        sparse_budget=args.budget,
        rank=160,
        chunk_size=8,
        router_centroids=8,
        router_centroid_method="self_lse_adaptive_iso",
        router_split_fraction=args.mean_components - 1.0,
        self_lse_temperatures=(1.0,),
        quest_prefix_tokens=32,
        quest_recent_tokens=32,
        group_reduce="max",
        streaming_offload=True,
        streaming_gather_backend="torch",
        streaming_router_backend="torch",
        streaming_refine_factor=1.0,
        streaming_max_components=8,
        streaming_compact_metadata=True,
        streaming_center_bits=8,
    )
    cache = llm.kv_cache
    wanted_layers = {layer for layer, _ in locations}
    prompt_queries: dict[int, torch.Tensor] = {}
    ranges: dict[int, tuple[int, int]] = {}
    layer_counter = 0

    original_rope = llm.apply_rotary_pos_emb
    original_logits = cache._block_logits

    def wrapped_rope(query, key, position_ids):
        nonlocal layer_counter
        query, key = original_rope(query, key, position_ids)
        if query.shape[-2] > 1:
            layer = layer_counter
            layer_counter += 1
            if layer in wanted_layers:
                prompt_queries[layer] = reduce_gqa(
                    query[..., -args.prompt_window:, :].detach(),
                    cache.num_key_value_heads,
                ).to(torch.bfloat16).cpu()
        return query, key

    def wrapped_logits(layer_idx, query_states, first_block, last_block):
        query, logits = original_logits(
            layer_idx, query_states, first_block, last_block
        )
        if layer_idx in wanted_layers:
            ranges[layer_idx] = (int(first_block), int(last_block))
        return query, logits

    llm.apply_rotary_pos_emb = wrapped_rope
    cache._block_logits = wrapped_logits
    input_ids = torch.tensor(
        llm.tokenizer.encode(row["input"], add_special_tokens=False),
        device="cuda:0", dtype=torch.long,
    )[None]
    logits = llm.prefill(input_ids)
    token = sample_token(logits[:, -1], temperature=0.0, top_p=1.0, top_k=50)
    cache.H2D()
    llm.inference(token, llm.get_ctx(token))
    llm.apply_rotary_pos_emb = original_rope
    cache._block_logits = original_logits

    trace = torch.load(args.trace, map_location="cpu", weights_only=False)
    trace_by_layer: dict[int, list[dict]] = {}
    for record in trace:
        if record["layer"] in wanted_layers:
            trace_by_layer.setdefault(record["layer"], []).append(record)
    for records in trace_by_layer.values():
        records.sort(key=lambda record: record["step"])

    select_blocks = args.budget // cache.block_size
    records_out: list[dict] = []
    for layer, head in sorted(locations):
        first, last = ranges[layer]
        n_blocks = last - first
        start, stop = first * cache.block_size, last * cache.block_size
        key = cache.k_cache[layer, 0, head, start:stop].to("cuda:0").view(
            1, 1, n_blocks, cache.block_size, cache.head_dim
        )
        prompt_window = prompt_queries[layer][:, head:head + 1].to("cuda:0")
        banks = causal_banks(prompt_window)

        future_full = torch.cat(
            [record["query"] for record in trace_by_layer[layer]], dim=-2
        )
        future = reduce_gqa(
            future_full, cache.num_key_value_heads
        )[:, head:head + 1].to("cuda:0")
        banks["oracle_future_all"] = future
        banks["oracle_future4"] = future[..., :4, :]
        if bank_names:
            banks = {name: bank for name, bank in banks.items() if name in bank_names}

        group = cache.num_key_value_groups
        future_individual = torch.cat(
            [record["query"] for record in trace_by_layer[layer]], dim=-2
        ).view(
            1, cache.num_key_value_heads, group, -1, cache.head_dim
        )[:, head:head + 1].float().to("cuda:0")
        exact_token_logits = torch.einsum(
            "bhgtd,bhnsd->bhgtns", future_individual, key.float()
        ) / math.sqrt(cache.head_dim)
        exact_block_lse = torch.logsumexp(exact_token_logits, dim=-1)
        exact_probability = torch.softmax(exact_block_lse, dim=-1).mean(2)[0, 0]
        oracle_ids = exact_probability.topk(select_blocks, dim=-1).indices
        oracle_evidence_recall = None
        oracle_max_gqa_evidence_recall = None
        oracle_peak_evidence_recall = None
        oracle_qmean_evidence_recall = None
        relative_evidence = torch.tensor(
            sorted(block - first for block in evidence_blocks
                   if first <= block < last),
            device=oracle_ids.device,
        )
        if relative_evidence.numel():
            def evidence_recall_for(ids: torch.Tensor) -> torch.Tensor:
                return torch.stack([
                    torch.isin(relative_evidence, ids[t]).float().mean()
                    for t in range(ids.shape[0])
                ])

            oracle_evidence_recall = torch.stack([
                torch.isin(relative_evidence, oracle_ids[t]).float().mean()
                for t in range(oracle_ids.shape[0])
            ])
            # Shared-KV selection policies that retain a block whenever any
            # GQA query head sees either large normalized mass or one sharp
            # token peak.  These distinguish evidence dilution across GQA
            # heads/tokens from centroid representation error.
            per_gqa_probability = torch.softmax(
                exact_block_lse, dim=-1
            )[0, 0]
            max_gqa_ids = per_gqa_probability.amax(0).topk(
                select_blocks, dim=-1
            ).indices
            peak_ids = exact_token_logits[0, 0].amax(dim=(0, -1)).topk(
                select_blocks, dim=-1
            ).indices
            qmean_logits = torch.einsum(
                "bhtd,bhnsd->bhtns", future, key.float()
            ) / math.sqrt(cache.head_dim)
            qmean_lse = torch.logsumexp(qmean_logits, dim=-1)[0, 0]
            qmean_ids = qmean_lse.topk(select_blocks, dim=-1).indices
            oracle_max_gqa_evidence_recall = evidence_recall_for(max_gqa_ids)
            oracle_peak_evidence_recall = evidence_recall_for(peak_ids)
            oracle_qmean_evidence_recall = evidence_recall_for(qmean_ids)

        for name, bank in banks.items():
            for objective in objectives:
                fit_kwargs = {
                    "objective": objective,
                    "tail_fraction": args.tail_fraction,
                }
                if objective in {"topk_hinge", "topk_hinge_cvar"}:
                    fit_kwargs["selection_fraction"] = min(
                        1.0, select_blocks / n_blocks
                    )
                risk, path, _ = fit_observed_query_mass_paths(
                    key, bank, **fit_kwargs
                )
                extra = int(round((args.mean_components - 1.0) * n_blocks))
                counts = _concave_marginal_counts(risk, extra)
                centers, log_counts, _ = _padded_adaptive_centroids(
                    key, path, counts, slot_count=cache.block_size
                )
                approximate = torch.einsum(
                    "bhtd,bhnrd->bhtnr", future, centers.float()
                ) / math.sqrt(cache.head_dim)
                approximate = torch.logsumexp(
                    approximate + log_counts.float()[:, :, None], dim=-1
                )[0, 0]
                selected = approximate.topk(select_blocks, dim=-1).indices
                chosen_mass = exact_probability.gather(-1, selected).sum(-1)
                overlap = torch.stack([
                    torch.isin(selected[t], oracle_ids[t]).float().mean()
                    for t in range(selected.shape[0])
                ])
                evidence_recall = None
                if relative_evidence.numel():
                    evidence_recall = torch.stack([
                        torch.isin(
                            relative_evidence, selected[t]
                        ).float().mean()
                        for t in range(selected.shape[0])
                    ])
                record = {
                    "layer": layer,
                    "head": head,
                    "method": f"{name}:{objective}",
                    "steps": int(selected.shape[0]),
                    "mean_components": float(counts.float().mean()),
                    "mass_mean": float(chosen_mass.mean()),
                    "mass_p10": float(chosen_mass.quantile(0.1)),
                    "mass_first": float(chosen_mass[0]),
                    "mass_last": float(chosen_mass[-1]),
                    "overlap_mean": float(overlap.mean()),
                    "overlap_p10": float(overlap.quantile(0.1)),
                    "evidence_recall_mean": (
                        float(evidence_recall.mean())
                        if evidence_recall is not None else None
                    ),
                    "evidence_recall_min": (
                        float(evidence_recall.min())
                        if evidence_recall is not None else None
                    ),
                    "evidence_recall_zero_steps": (
                        int((evidence_recall == 0).sum())
                        if evidence_recall is not None else None
                    ),
                    "oracle_evidence_recall_mean": (
                        float(oracle_evidence_recall.mean())
                        if oracle_evidence_recall is not None else None
                    ),
                    "oracle_evidence_recall_min": (
                        float(oracle_evidence_recall.min())
                        if oracle_evidence_recall is not None else None
                    ),
                    "oracle_evidence_recall_zero_steps": (
                        int((oracle_evidence_recall == 0).sum())
                        if oracle_evidence_recall is not None else None
                    ),
                    "oracle_max_gqa_evidence_recall_mean": (
                        float(oracle_max_gqa_evidence_recall.mean())
                        if oracle_max_gqa_evidence_recall is not None else None
                    ),
                    "oracle_max_gqa_evidence_recall_min": (
                        float(oracle_max_gqa_evidence_recall.min())
                        if oracle_max_gqa_evidence_recall is not None else None
                    ),
                    "oracle_peak_evidence_recall_mean": (
                        float(oracle_peak_evidence_recall.mean())
                        if oracle_peak_evidence_recall is not None else None
                    ),
                    "oracle_peak_evidence_recall_min": (
                        float(oracle_peak_evidence_recall.min())
                        if oracle_peak_evidence_recall is not None else None
                    ),
                    "oracle_qmean_evidence_recall_mean": (
                        float(oracle_qmean_evidence_recall.mean())
                        if oracle_qmean_evidence_recall is not None else None
                    ),
                    "oracle_qmean_evidence_recall_min": (
                        float(oracle_qmean_evidence_recall.min())
                        if oracle_qmean_evidence_recall is not None else None
                    ),
                }
                records_out.append(record)
                print(json.dumps(record), flush=True)
                del risk, path, centers, approximate

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "sample": args.sample_index,
        "trace": str(args.trace),
        "prompt_window": args.prompt_window,
        "records": records_out,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
