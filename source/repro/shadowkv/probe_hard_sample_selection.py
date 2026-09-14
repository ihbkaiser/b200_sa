#!/usr/bin/env python3
"""Audit answer-span selection in a streaming-Quest or ShadowKV failure run."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))


def load_row(path: Path, index: int) -> dict:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(index)


def flatten(value) -> str:
    while isinstance(value, list):
        value = value[0]
    return str(value)


def spans(sequence: list[int], pattern: list[int]) -> list[tuple[int, int]]:
    width = len(pattern)
    return [
        (start, start + width)
        for start in range(len(sequence) - width + 1)
        if sequence[start : start + width] == pattern
    ]


def answer_spans(tokenizer, ids: list[int], answer: str) -> list[tuple[int, int]]:
    found = set()
    for text in (answer, " " + answer, "\n" + answer, ": " + answer):
        found.update(spans(ids, tokenizer.encode(text, add_special_tokens=False)))
    return sorted(found)


def stop(llm, token: torch.Tensor) -> bool:
    value = int(token[0, 0])
    text = llm.tokenizer.decode([value])
    return value == llm.tokenizer.eos_token_id or text in {
        "<|eot_id|>", "<|im_end|>", "<|endoftext|>", "<|end|>"
    } or value in {151329, 151336, 151338}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument(
        "--method",
        choices=(
            "quest_streaming",
            "shadowkv",
            "shadowkv_quill",
            "shadowkv_centroid_lse",
        ),
        required=True,
    )
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument(
        "--router-split-fraction",
        type=float,
        default=0.25,
        help="adaptive centroid extra-component budget; mean r is 1+fraction",
    )
    parser.add_argument(
        "--router-variant",
        choices=("adaptive_iso", "exact_lse"),
        default="adaptive_iso",
        help="diagnostic centroid representation; exact_lse keeps all 8 keys",
    )
    parser.add_argument("--max-decode-steps", type=int, default=32)
    parser.add_argument("--force-evidence", choices=("none", "all", "tail"), default="none")
    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="save per-call queries and selected block ids for causal diagnostics",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from models import choose_model_class
    from models.tensor_op import sample_token

    row = load_row(args.dataset, args.sample_index)
    llm_class = choose_model_class(args.model)
    kwargs = dict(
        model_name=args.model,
        batch_size=1,
        device="cuda:0",
        max_length=32768 + args.max_decode_steps + 2048,
        attn_mode=args.method,
        dtype=torch.bfloat16,
        sparse_budget=args.budget,
        rank=160,
        chunk_size=8,
    )
    if args.method == "quest_streaming":
        kwargs.update(
            page_size=16, dense_layers=2, group_reduce="max",
            quest_recent_tokens=32,
        )
    elif args.method == "shadowkv_centroid_lse":
        if args.router_variant == "adaptive_iso":
            kwargs.update(
                router_centroids=8,
                router_centroid_method="self_lse_adaptive_iso",
                router_split_fraction=args.router_split_fraction,
                self_lse_temperatures=(1.0,),
                shadow_outlier_chunks=48,
            )
        else:
            kwargs.update(
                router_centroids=8,
                router_centroid_method="kmeans",
                router_split_fraction=1.0,
                self_lse_temperatures=(1.0,),
                shadow_outlier_chunks=48,
            )
    llm = llm_class(**kwargs)
    token_list = llm.tokenizer.encode(row["input"], add_special_tokens=False)
    input_ids = torch.tensor(token_list, device="cuda", dtype=torch.long)[None]
    answer = flatten(row["outputs"])
    found = answer_spans(llm.tokenizer, token_list, answer)
    if not found:
        raise RuntimeError(f"could not locate {answer!r}")
    evidence = sorted({position for start, end in found for position in range(start, end)})
    evidence_tensor = torch.tensor(evidence, dtype=torch.long)
    unit_size = 16 if args.method == "quest_streaming" else 8
    evidence_units = sorted({position // unit_size for position in evidence})
    if args.force_evidence == "tail":
        evidence_units = evidence_units[-1:]
    elif args.force_evidence == "none":
        evidence_units = []

    captured: list[tuple[int, torch.Tensor]] = []
    captured_queries: list[tuple[int, torch.Tensor]] = []
    original = llm.kv_cache.get_retrieval_position_ids

    def wrapped(layer_idx, query_states):
        if args.save_trace:
            captured_queries.append(
                (int(layer_idx), query_states.detach().float().cpu())
            )
        selected = original(layer_idx, query_states)
        if evidence_units:
            unit_count = selected.shape[-1] // unit_size
            forced_selected = selected.clone()
            offsets = torch.arange(unit_size, device=selected.device)
            for head in range(llm.num_key_value_heads):
                forced = evidence_units
                if args.method.startswith("shadowkv"):
                    # Outlier chunks are already permanently resident.  Do not
                    # duplicate them in the dynamic slot, which would count
                    # their attention twice.
                    rest = set(
                        int(value)
                        for value in llm.kv_cache.k_landmark_idx[
                            layer_idx, 0, head
                        ].tolist()
                    )
                    forced = [unit for unit in forced if unit in rest]
                existing = (
                    selected[0, head].reshape(unit_count, unit_size)[:, 0] // unit_size
                ).tolist()
                merged = list(forced)
                merged.extend(unit for unit in existing if unit not in set(forced))
                merged = merged[:unit_count]
                units = torch.tensor(merged, device=selected.device, dtype=torch.long)
                forced_selected[0, head] = (
                    units[:, None] * unit_size + offsets[None]
                ).reshape(-1)
            selected = forced_selected
        captured.append((int(layer_idx), selected.detach().cpu()))
        return selected

    llm.kv_cache.get_retrieval_position_ids = wrapped
    logits = llm.prefill(input_ids)
    logit_trace = []
    values, indices = logits[:, -1].float().topk(10, dim=-1)
    logit_trace.append(
        {
            "source": "prefill",
            "ids": indices[0].cpu().tolist(),
            "logits": values[0].cpu().tolist(),
        }
    )
    token = sample_token(logits[:, -1], temperature=0.0, top_p=1.0, top_k=50)
    generated = [int(token[0, 0])]
    llm.kv_cache.H2D()
    for _ in range(args.max_decode_steps - 1):
        logits = llm.inference(token, llm.get_ctx(token))
        values, indices = logits[:, -1].float().topk(10, dim=-1)
        logit_trace.append(
            {
                "source": "decode",
                "ids": indices[0].cpu().tolist(),
                "logits": values[0].cpu().tolist(),
            }
        )
        token = sample_token(logits[:, -1], temperature=0.0, top_p=1.0, top_k=50)
        generated.append(int(token[0, 0]))
        if stop(llm, token):
            break
    llm.kv_cache.get_retrieval_position_ids = original

    call_by_layer = {layer: 0 for layer in range(llm.num_layers)}
    records = []
    position_stats = {
        position: {"hits": 0, "total": 0, "sparse_hits": 0, "sparse_total": 0}
        for position in evidence
    }
    evidence_blocks8 = {position // 8 for position in evidence}
    for layer, selected in captured:
        step = call_by_layer[layer]
        call_by_layer[layer] += 1
        # Quest does not invoke retrieval for its two dense layers.
        for head in range(llm.num_key_value_heads):
            dynamic = set(int(value) for value in selected[0, head].tolist())
            fixed: set[int] = set()
            if args.method.startswith("shadowkv"):
                rest = set(int(value) for value in llm.kv_cache.k_landmark_idx[layer, 0, head].tolist())
                all_chunk = set(range(int(llm.kv_cache.chunks)))
                outlier_chunks = all_chunk - rest
                for chunk in outlier_chunks:
                    fixed.update(range(chunk * 8, chunk * 8 + 8))
                fixed.update(range(llm.kv_cache.chunks * 8, len(token_list)))
            elif args.method == "quest_streaming":
                for start, end in llm.kv_cache.block_state[layer].exact_ranges:
                    fixed.update(range(start, end))
            retained = dynamic | fixed
            evidence_component_counts = []
            if args.method == "shadowkv_centroid_lse":
                candidate = llm.kv_cache.k_landmark_idx[layer, 0, head]
                packed_counts = llm.kv_cache.router_component_count[layer]
                counts = (
                    packed_counts[0, head]
                    if packed_counts is not None
                    else torch.full_like(candidate, 8)
                )
                for block in evidence_blocks8:
                    location = (candidate == block).nonzero(as_tuple=False)
                    if location.numel():
                        evidence_component_counts.append(
                            int(counts[int(location[0, 0])].item())
                        )
            for position in evidence:
                hit = int(position in retained)
                position_stats[position]["hits"] += hit
                position_stats[position]["total"] += 1
                position_stats[position]["sparse_hits"] += hit
                position_stats[position]["sparse_total"] += 1
            hits = sum(position in retained for position in evidence)
            block_hits = sum(any(position in retained for position in range(block * 8, block * 8 + 8)) for block in evidence_blocks8)
            records.append(
                {
                    "method": args.method,
                    "budget": args.budget,
                    "sample_index": args.sample_index,
                    "step": step,
                    "layer": layer,
                    "kv_head": head,
                    "dynamic_tokens": len(dynamic),
                    "fixed_tokens": len(fixed),
                    "evidence_tokens": len(evidence),
                    "evidence_token_recall": hits / len(evidence),
                    "evidence_any": float(hits > 0),
                    "evidence_all": float(hits == len(evidence)),
                    "evidence_block_recall": block_hits / len(evidence_blocks8),
                    "evidence_candidate_blocks": len(evidence_component_counts),
                    "evidence_component_mean": (
                        sum(evidence_component_counts) / len(evidence_component_counts)
                        if evidence_component_counts else None
                    ),
                    "evidence_component_min": (
                        min(evidence_component_counts)
                        if evidence_component_counts else None
                    ),
                    "evidence_component_max": (
                        max(evidence_component_counts)
                        if evidence_component_counts else None
                    ),
                }
            )

    # Dense streaming-Quest layers are explicit rows rather than silently omitted.
    if args.method == "quest_streaming":
        decode_steps = max(call_by_layer.values())
        for step in range(decode_steps):
            for layer in range(llm.kv_cache.dense_layers):
                for head in range(llm.num_key_value_heads):
                    for position in evidence:
                        position_stats[position]["hits"] += 1
                        position_stats[position]["total"] += 1
                    records.append(
                        {
                            "method": args.method,
                            "budget": args.budget,
                            "sample_index": args.sample_index,
                            "step": step,
                            "layer": layer,
                            "kv_head": head,
                            "dynamic_tokens": len(token_list),
                            "fixed_tokens": 0,
                            "evidence_tokens": len(evidence),
                            "evidence_token_recall": 1.0,
                            "evidence_any": 1.0,
                            "evidence_all": 1.0,
                            "evidence_block_recall": 1.0,
                        }
                    )

    frame = pd.DataFrame(records).sort_values(["step", "layer", "kv_head"])
    per_step = (
        frame.groupby("step", as_index=False)
        .agg(
            evidence_token_recall=("evidence_token_recall", "mean"),
            evidence_any=("evidence_any", "mean"),
            evidence_all=("evidence_all", "mean"),
            evidence_block_recall=("evidence_block_recall", "mean"),
        )
    )
    position_frame = pd.DataFrame(
        [
            {
                "position": position,
                "relative_to_evidence_start": position - min(evidence),
                "block8": position // 8,
                "offset8": position % 8,
                "token_id": token_list[position],
                "token_text": llm.tokenizer.decode([token_list[position]]),
                "retention": state["hits"] / state["total"],
                "sparse_layer_retention": state["sparse_hits"] / state["sparse_total"],
            }
            for position, state in sorted(position_stats.items())
        ]
    )
    block_frame = (
        position_frame.groupby("block8", as_index=False)
        .agg(
            positions=("position", "count"),
            first_relative=("relative_to_evidence_start", "min"),
            last_relative=("relative_to_evidence_start", "max"),
            retention=("retention", "mean"),
            sparse_layer_retention=("sparse_layer_retention", "mean"),
        )
    )
    summary = {
        "method": args.method,
        "router_variant": args.router_variant,
        "budget": args.budget,
        "router_split_fraction": args.router_split_fraction,
        "force_evidence": args.force_evidence,
        "sample_index": args.sample_index,
        "answer": answer,
        "generated": llm.tokenizer.decode(generated, skip_special_tokens=True),
        "generated_ids": generated,
        "logit_trace": logit_trace,
        "generated_contains_answer": answer.lower() in llm.tokenizer.decode(generated, skip_special_tokens=True).lower(),
        "prompt_tokens": len(token_list),
        "decode_queries": int(frame.step.nunique()),
        "evidence_positions": evidence,
        "evidence_blocks8": sorted(evidence_blocks8),
        "mean_evidence_token_recall": float(frame.evidence_token_recall.mean()),
        "mean_evidence_any": float(frame.evidence_any.mean()),
        "mean_evidence_all": float(frame.evidence_all.mean()),
        "mean_evidence_block_recall": float(frame.evidence_block_recall.mean()),
        "first_step_evidence_any_below_half": (
            int(per_step.loc[per_step.evidence_any < 0.5, "step"].iloc[0])
            if (per_step.evidence_any < 0.5).any()
            else None
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output / "per_layer_head_step.parquet", index=False)
    per_step.to_csv(args.output / "per_step.csv", index=False)
    position_frame.to_csv(args.output / "per_evidence_position.csv", index=False)
    block_frame.to_csv(args.output / "per_evidence_block.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.save_trace:
        if len(captured_queries) != len(captured):
            raise RuntimeError("query and selection traces are misaligned")
        trace = []
        for (query_layer, query), (selected_layer, selected) in zip(
            captured_queries, captured
        ):
            if query_layer != selected_layer:
                raise RuntimeError("query and selection layers are misaligned")
            trace.append(
                {
                    "layer": query_layer,
                    "query": query,
                    "selected_blocks": (
                        selected[..., ::unit_size] // unit_size
                    ).to(torch.int32),
                }
            )
        torch.save(trace, args.output / "trace.pt")
    if args.method == "shadowkv_quill":
        torch.save(
            {
                "candidate_positions": torch.stack(
                    llm.kv_cache.router_candidate_positions, dim=0
                ),
                "tokens": len(token_list),
                "exact_fraction": llm.kv_cache.exact_fraction,
            },
            args.output / "quill_candidates.pt",
        )
    print(per_step.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
