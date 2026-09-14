#!/usr/bin/env python3
"""Measure CWE attention geometry on the model's true full-attention trajectory."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
SHADOW = REPO / "ShadowKV"
sys.path.insert(0, str(SHADOW))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_cwe_streaming import load_row, source_occurrences


def shared_oracle_mass(
    probability: torch.Tensor,
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    budget: int,
    block_size: int,
    prefix_tokens: int = 32,
    recent_tokens: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mass retained by exact shared-GQA top blocks at a token budget."""
    total = logits.shape[-1]
    sealed = total // block_size
    active = min(sealed, max(0, total - recent_tokens) // block_size)
    first = min(prefix_tokens // block_size, active)
    candidate = logits[..., first * block_size : active * block_size]
    candidate = candidate.view(*candidate.shape[:-1], active - first, block_size)
    block_logits = torch.logsumexp(candidate, dim=-1)
    block_probability = torch.softmax(block_logits, dim=-1)
    score = block_probability.sum(dim=-2).amax(dim=2)
    count = min(budget // block_size, score.shape[-1])
    selected = torch.topk(score, count, dim=-1).indices + first

    batch, kv_heads = selected.shape[:2]
    selected_mask = torch.zeros(
        batch, kv_heads, total, device=logits.device, dtype=torch.bool
    )
    if prefix_tokens:
        selected_mask[..., : min(prefix_tokens, total)] = True
    suffix_start = active * block_size
    selected_mask[..., suffix_start:] = True
    offsets = torch.arange(block_size, device=logits.device)
    positions = (selected[..., None] * block_size + offsets).flatten(-2)
    selected_mask.scatter_(-1, positions, True)

    expanded = selected_mask[:, :, None, None, :]
    retained = probability.masked_fill(~expanded, 0).sum(-1)
    retained_target = probability.masked_fill(
        ~(expanded & target_mask[None, None, None, None, :]), 0
    ).sum(-1)
    full_target = probability[..., target_mask].sum(-1).clamp_min(1e-30)
    return retained.mean(dim=(-1, -2)), (retained_target / full_target).mean(dim=(-1, -2))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from data.metrics import multi_words
    from models import choose_model_class

    row = load_row(args.dataset, args.sample_index)
    llm_class = choose_model_class(args.model)
    llm = llm_class(
        model_name=args.model,
        batch_size=1,
        device="cuda:0",
        max_length=32768 + args.gen_len + 2048,
        attn_mode="full",
        dtype=torch.bfloat16,
    )
    encoded = llm.tokenizer(
        row["input"], add_special_tokens=False, return_offsets_mapping=True
    )
    input_ids = torch.tensor(encoded.input_ids, device="cuda:0", dtype=torch.long)[None]
    answers = [str(word).lower() for word in row["outputs"]]
    _, _, _, target_positions, source_positions = source_occurrences(
        row["input"], encoded.offset_mapping, 8, answers
    )

    pending_query: torch.Tensor | None = None
    current_layer = -1
    steps = [0 for _ in range(llm.num_layers)]
    rows: list[dict] = []

    original_rope = llm.apply_rotary_pos_emb
    original_update = llm.kv_cache.update_kv_cache
    original_layer = llm.layer_compute

    def wrapped_layer(buffer, layer_idx, hidden_states, position_ids):
        nonlocal current_layer
        current_layer = layer_idx
        return original_layer(buffer, layer_idx, hidden_states, position_ids)

    def wrapped_rope(query, key, position_ids):
        nonlocal pending_query
        query, key = original_rope(query, key, position_ids)
        if query.shape[-2] == 1:
            pending_query = query.detach()
        return query, key

    def wrapped_update(new_key, new_value, layer_idx):
        nonlocal pending_query
        key, value = original_update(new_key, new_value, layer_idx)
        if new_key.shape[-2] != 1:
            return key, value
        if pending_query is None or current_layer != layer_idx:
            raise RuntimeError("failed to associate the decode query with its layer")

        total = key.shape[-2]
        query = pending_query.view(
            1, llm.num_key_value_heads, llm.num_key_value_groups, 1, llm.head_dim
        ).float()
        logits = torch.einsum("bhgqd,bhtd->bhgqt", query, key.float())
        logits /= math.sqrt(llm.head_dim)
        probability = torch.softmax(logits, dim=-1)

        target_mask = torch.zeros(total, device=logits.device, dtype=torch.bool)
        source_mask = torch.zeros_like(target_mask)
        target_ids = [position for position in target_positions if position < total]
        source_ids = [position for position in source_positions if position < total]
        if target_ids:
            target_mask[target_ids] = True
        if source_ids:
            source_mask[source_ids] = True
        distractor_mask = source_mask & ~target_mask

        target_mass = probability[..., target_mask].sum(-1)
        distractor_mass = probability[..., distractor_mask].sum(-1)
        flat = probability.flatten(-2)
        entropy = -(flat * flat.clamp_min(1e-30).log()).sum(-1)
        top1 = flat.amax(-1)
        top8 = torch.topk(flat, min(8, total), dim=-1).values.sum(-1)
        peak_is_target = target_mask[flat.argmax(-1)]
        c8_mass, c8_target = shared_oracle_mass(
            probability, logits, target_mask, budget=args.budget, block_size=8
        )
        c2_mass, c2_target = shared_oracle_mass(
            probability, logits, target_mask, budget=args.budget, block_size=2
        )

        target_count = max(1, int(target_mask.sum()))
        distractor_count = max(1, int(distractor_mask.sum()))
        for head in range(llm.num_key_value_heads):
            rows.append(
                {
                    "sample": args.sample_index,
                    "step": steps[layer_idx],
                    "layer": layer_idx,
                    "kv_head": head,
                    "target_mass": target_mass[0, head].mean().item(),
                    "distractor_word_mass": distractor_mass[0, head].mean().item(),
                    "target_vs_distractor_density_ratio": (
                        (target_mass[0, head] / target_count)
                        / (distractor_mass[0, head] / distractor_count).clamp_min(1e-30)
                    ).mean().item(),
                    "top1_token_mass": top1[0, head].mean().item(),
                    "top8_token_mass": top8[0, head].mean().item(),
                    "effective_attention_tokens": entropy[0, head].exp().mean().item(),
                    "peak_token_is_target_rate": peak_is_target[0, head].float().mean().item(),
                    "c8_oracle_full_mass_retained": c8_mass[0, head].item(),
                    "c8_oracle_target_mass_retained": c8_target[0, head].item(),
                    "c2_oracle_full_mass_retained": c2_mass[0, head].item(),
                    "c2_oracle_target_mass_retained": c2_target[0, head].item(),
                }
            )
        steps[layer_idx] += 1
        pending_query = None
        return key, value

    llm.layer_compute = wrapped_layer
    llm.apply_rotary_pos_emb = wrapped_rope
    llm.kv_cache.update_kv_cache = wrapped_update
    prediction = llm.generate(input_ids, gen_len=args.gen_len, temperature=0.0)[0]

    args.output.mkdir(parents=True, exist_ok=True)
    calls = pd.DataFrame(rows)
    calls.to_csv(args.output / "full_attention_calls.csv", index=False)
    by_layer = calls.groupby("layer", as_index=False).mean(numeric_only=True)
    by_layer.to_csv(args.output / "by_layer.csv", index=False)
    summary = {
        "sample": args.sample_index,
        "prompt_tokens": int(input_ids.shape[-1]),
        "prediction": prediction,
        "ground_truth": answers,
        "score": float(multi_words(prediction, answers)),
        "calls": len(calls),
        **calls.drop(columns=["sample", "step", "layer", "kv_head"]).mean().to_dict(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(by_layer.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
