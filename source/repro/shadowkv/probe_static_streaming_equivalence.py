#!/usr/bin/env python3
"""Dump the first decode retrieval decision for static/streaming ours.

The probe intentionally generates only two tokens: the first comes from the
prefill logits and the second triggers exactly one retrieval call per layer.
No generated block can be sealed at block size eight, so any disagreement is
already present in the post-prefill cache state or routing implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
SHADOW = REPO / "ShadowKV"
sys.path.insert(0, str(SHADOW))


def load_row(path: Path, index: int) -> dict:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--mode", choices=("static", "streaming"), required=True)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument(
        "--streaming-include-prefix-normalizer",
        action="store_true",
        help="score prefix blocks in the per-query-head softmax, then mask them",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from models import choose_model_class

    row = load_row(args.dataset, args.sample_index)
    attn_mode = (
        "adaptive_centroid_lse_prefix4"
        if args.mode == "static"
        else "adaptive_centroid_lse_streaming_prefix4"
    )
    llm_class = choose_model_class(args.model)
    llm = llm_class(
        model_name=args.model,
        batch_size=1,
        device="cuda:0",
        max_length=32768 + 64,
        attn_mode=attn_mode,
        dtype=torch.bfloat16,
        sparse_budget=args.budget,
        rank=160,
        chunk_size=8,
        router_centroids=8,
        router_centroid_method="self_lse_adaptive_iso",
        router_split_fraction=0.25,
        self_lse_temperatures=(1.0,),
        quest_recent_tokens=32,
        group_reduce="max",
    )
    cache = llm.kv_cache
    original = cache.get_retrieval_position_ids
    first_ids: dict[int, torch.Tensor] = {}
    first_scores: dict[int, torch.Tensor] = {}
    first_queries: dict[int, torch.Tensor] = {}

    def static_scores(layer_idx: int, query_states: torch.Tensor) -> torch.Tensor:
        query = query_states.view(
            cache.batch_size,
            cache.num_key_value_heads,
            cache.num_key_value_groups,
            cache.incoming_q_len,
            cache.head_dim,
        )
        component_logits = torch.einsum(
            "bhgqd,bhcd->bhgqc", query, cache.router_centroids[layer_idx]
        ).float() / math.sqrt(cache.head_dim)
        component_logits += cache.router_log_counts[layer_idx].float()[
            :, :, None, None, :
        ]
        query_norm2 = query.float().square().sum(-1) / cache.head_dim
        component_logits += query_norm2[..., None] * (
            cache.router_isotropic_alpha[layer_idx][
                :, :, None, None, :
            ].float()
        )
        component_block = cache.router_component_block[layer_idx]
        index = component_block[:, :, None, None, :].expand_as(component_logits)
        n_blocks = cache.k_landmark_idx[layer_idx].shape[-1]
        block_max = torch.full(
            (*component_logits.shape[:-1], n_blocks),
            float("-inf"),
            device=component_logits.device,
            dtype=component_logits.dtype,
        )
        block_max.scatter_reduce_(
            -1, index, component_logits, reduce="amax", include_self=True
        )
        centered = component_logits - block_max.gather(-1, index)
        block_sum = torch.zeros_like(block_max)
        block_sum.scatter_add_(-1, index, centered.exp())
        block_logits = block_max + block_sum.log()
        block_logits = cache._mask_router_block_logits(layer_idx, block_logits)
        scores = torch.softmax(block_logits, -1, dtype=torch.float32).to(cache.dtype)
        scores = scores.sum(-2)
        if cache.num_key_value_groups > 1:
            scores = scores.amax(-2)
        return scores

    def wrapped(layer_idx: int, query_states: torch.Tensor) -> torch.Tensor:
        if args.mode == "streaming" and args.streaming_include_prefix_normalizer:
            state = cache.block_state[layer_idx]
            first, last = state.candidate_block_range
            scores = cache._score_blocks(layer_idx, query_states, 0, last)
            scores[..., :first] = float("-inf")
            block_ids = torch.topk(
                scores, k=cache.select_blocks, dim=-1
            ).indices
            offsets = torch.arange(cache.block_size, device=scores.device)
            ids = (
                block_ids.unsqueeze(-1) * cache.block_size + offsets
            ).reshape(cache.batch_size, cache.num_key_value_heads, -1)
        else:
            ids = original(layer_idx, query_states)
        if layer_idx not in first_ids:
            first_ids[layer_idx] = ids.detach().cpu()
            first_queries[layer_idx] = query_states.detach().cpu()
            if args.mode == "static":
                score = static_scores(layer_idx, query_states)
            else:
                first, last = cache.block_state[layer_idx].candidate_block_range
                score = cache._score_blocks(
                    layer_idx, query_states, first, last
                )
            first_scores[layer_idx] = score.detach().cpu()
        return ids

    cache.get_retrieval_position_ids = wrapped
    input_ids = llm.tokenizer(
        row["input"], add_special_tokens=False, return_tensors="pt"
    ).input_ids.to("cuda:0")
    prediction = llm.generate(
        input_ids, gen_len=2, temperature=0.0, top_p=1.0, top_k=50
    )[0]

    components: dict[int, dict[str, torch.Tensor]] = {}
    if args.mode == "static":
        for layer, counts in enumerate(cache.router_component_count):
            if counts is None:
                continue
            components[layer] = {
                "block_ids": cache.k_landmark_idx[layer].detach().cpu(),
                "counts": counts.detach().cpu(),
            }
    else:
        for layer, state in enumerate(cache.block_state):
            first, last = state.candidate_block_range
            components[layer] = {
                "block_ids": torch.arange(first, last)[None, None].expand(
                    cache.batch_size, cache.num_key_value_heads, -1
                ),
                "counts": cache.component_count[
                    layer, :, :, first:last
                ].detach().cpu(),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mode": args.mode,
            "streaming_include_prefix_normalizer": (
                args.streaming_include_prefix_normalizer
            ),
            "sample_index": args.sample_index,
            "prompt_tokens": int(input_ids.shape[-1]),
            "prediction": prediction,
            "retrieval_ids": first_ids,
            "router_scores": first_scores,
            "queries": first_queries,
            "components": components,
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "sample": args.sample_index,
                "prompt_tokens": int(input_ids.shape[-1]),
                "prediction": prediction,
                "layers": len(first_ids),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
