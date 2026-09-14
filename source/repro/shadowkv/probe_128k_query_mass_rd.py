#!/usr/bin/env python3
"""Targeted 128K test of prompt-query mass rate--distortion routing.

The model uses the deployed CPU-offloaded streaming cache.  A small bank of
post-RoPE queries already observed near the end of prefill fits a nested 1..8
centroid path and allocates the same global number of centers as the deployed
self-K router.  Future decode queries are used only for evaluation.
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

from diagnose_router_mass_full_trace import (  # noqa: E402
    _concave_marginal_counts,
    fit_agglomerative_query_mass_paths,
)
from models.adaptive_centroid_streaming_cache import (  # noqa: E402
    _padded_adaptive_centroids,
)


def load_row(path: Path, index: int) -> dict:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(index)


def parse_locations(spec: str) -> set[tuple[int, int]]:
    output = set()
    for item in spec.split(","):
        layer, head = item.split(":")
        output.add((int(layer), int(head)))
    return output


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--datalen", type=int, default=131072)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--mean-components", type=float, default=1.5)
    ap.add_argument("--query-bank-size", type=int, default=32)
    ap.add_argument("--query-window", type=int, default=512)
    ap.add_argument("--locations", default="5:6")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    locations = parse_locations(args.locations)

    os.environ["SHADOWKV_CENTER_PLACEMENT"] = "robust_trimmed"
    os.environ["SHADOWKV_CENTER_ALLOCATION"] = "tail_cvar"
    os.environ["SHADOWKV_ROBUST_TRIM_FRACTION"] = "0.25"
    os.environ["SHADOWKV_TAIL_CVAR_FRACTION"] = "0.25"
    os.environ["SHADOWKV_TAIL_GAP_CORRECTION_SCALE"] = "0.25"
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
    prefill_bank: dict[int, torch.Tensor] = {}
    decode_capture: dict[int, tuple[torch.Tensor, torch.Tensor, int, int]] = {}
    prefill_counter = 0

    original_rope = llm.apply_rotary_pos_emb
    def wrapped_rope(query, key, position_ids):
        nonlocal prefill_counter
        query, key = original_rope(query, key, position_ids)
        if query.shape[-2] > 1:
            layer = prefill_counter
            prefill_counter += 1
            if layer in wanted_layers:
                length = query.shape[-2]
                start = max(0, length - args.query_window)
                count = min(args.query_bank_size, length - start)
                index = torch.linspace(
                    start, length - 1, count, device=query.device
                ).round().long()
                prefill_bank[layer] = query[..., index, :].detach().to(
                    torch.bfloat16
                ).cpu()
        return query, key

    original_logits = cache._block_logits
    def wrapped_logits(layer_idx, query_states, first_block, last_block):
        query, logits = original_logits(
            layer_idx, query_states, first_block, last_block
        )
        if layer_idx in wanted_layers and layer_idx not in decode_capture:
            decode_capture[layer_idx] = (
                query_states.detach().to(torch.bfloat16).cpu(),
                logits.detach().float().cpu(),
                int(first_block), int(last_block),
            )
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

    select_blocks = args.budget // cache.block_size
    records = []
    for layer, head in sorted(locations):
        raw_query, deployed_logits, first, last = decode_capture[layer]
        n_blocks = last - first
        start, stop = first * cache.block_size, last * cache.block_size
        key = cache.k_cache[layer, 0, head, start:stop].to("cuda:0").view(
            1, 1, n_blocks, cache.block_size, cache.head_dim
        )
        bank_full = prefill_bank[layer].to("cuda:0")
        group = cache.num_key_value_groups
        bank = bank_full[:, head * group:(head + 1) * group]
        risk, path, _ = fit_agglomerative_query_mass_paths(key, bank)
        extra = int(round((args.mean_components - 1.0) * n_blocks))
        counts = _concave_marginal_counts(risk, extra)
        centers, log_counts, _ = _padded_adaptive_centroids(
            key, path, counts, slot_count=cache.block_size
        )

        q_full = raw_query.view(
            1, cache.num_key_value_heads, group, 1, cache.head_dim
        ).float().to("cuda:0")[:, head:head + 1]
        qmean = q_full.mean(2, keepdim=True)
        component = torch.einsum(
            "bhgqd,bhnrd->bhgqnr", qmean, centers.float()
        ) / math.sqrt(cache.head_dim)
        component = component + log_counts.float()[:, :, None, None]
        qmass_logits = torch.logsumexp(component, dim=-1).squeeze(2).squeeze(2)[0, 0]
        qmass_prob = torch.softmax(qmass_logits, dim=-1)

        token_logits = torch.einsum(
            "gd,nsd->gns", q_full[0, 0, :, 0], key[0, 0].float()
        ) / math.sqrt(cache.head_dim)
        exact_lse = torch.logsumexp(token_logits, dim=-1)
        exact_prob = torch.softmax(exact_lse, dim=-1).mean(0)
        deployed = deployed_logits[0, head].squeeze(1).squeeze(1).to("cuda:0")
        deployed_prob = torch.softmax(deployed, dim=-1)

        oracle_ids = exact_prob.topk(select_blocks).indices
        oracle_mask = torch.zeros(n_blocks, device="cuda:0", dtype=torch.bool)
        oracle_mask[oracle_ids] = True
        for name, probability in (
            ("deployed_self_k", deployed_prob),
            ("prompt_q_mass_rd", qmass_prob),
        ):
            ids = probability.topk(select_blocks).indices
            mask = torch.zeros_like(oracle_mask); mask[ids] = True
            records.append({
                "layer": layer, "head": head, "method": name,
                "mean_components": float(counts.float().mean()) if name == "prompt_q_mass_rd" else args.mean_components,
                "exact_mass": float(exact_prob[mask].sum()),
                "oracle_exact_mass": float(exact_prob[oracle_mask].sum()),
                "topb_overlap": float((mask & oracle_mask).sum() / select_blocks),
            })
        del key, risk, path, centers, token_logits

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "sample": args.sample_index,
        "query_bank": {"window": args.query_window, "size": args.query_bank_size},
        "records": records,
    }, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
