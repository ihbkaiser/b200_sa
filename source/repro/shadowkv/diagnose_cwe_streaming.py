#!/usr/bin/env python3
"""Diagnose CWE failures of the streaming adaptive-centroid router.

The probe executes the released campaign path, but evaluates the exact block
LSE and the deployed 1/2-centroid approximation on every *same-trajectory*
router call.  It records selection overlap, exact attention mass recovered by
each selection, approximation error, and whether exact-top blocks received a
second centroid.  No alternate query distribution or offline calibration is
introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
SHADOW = REPO / "ShadowKV"
sys.path.insert(0, str(SHADOW))

SOURCE_RE = re.compile(r"(?:^|\s)\d+\.\s+([^\s]+)")


def load_row(path: Path, index: int) -> dict:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(index)


def parse_int_set(spec: str, upper: int) -> set[int]:
    if not spec:
        return set()
    if spec == "all":
        return set(range(upper))
    result: set[int] = set()
    for field in spec.split(","):
        if "-" in field:
            left, right = map(int, field.split("-", 1))
            result.update(range(left, right + 1))
        else:
            result.add(int(field))
    if result and (min(result) < 0 or max(result) >= upper):
        raise ValueError(f"invalid subset {spec!r} for upper={upper}")
    return result


def source_occurrences(
    text: str,
    offsets: list[tuple[int, int]],
    block_size: int,
    answers: list[str],
):
    """Map the evaluated (not demonstration) word list back to token ids."""
    marker = "Below is a numbered list of words."
    body_start = text.rfind(marker)
    if body_start < 0:
        raise ValueError("CWE list marker not found")
    body_start += len(marker)
    body_stop = text.find("Question:", body_start)
    if body_stop < 0:
        raise ValueError("CWE question marker not found")
    body = text[body_start:body_stop]
    answer_set = set(answers)
    by_block: dict[int, Counter] = defaultdict(Counter)
    total = Counter()
    token_words: dict[int, set[str]] = defaultdict(set)
    target_positions: set[int] = set()
    source_positions: set[int] = set()
    for match in SOURCE_RE.finditer(body):
        raw = match.group(1)
        word = raw.strip(".,;:!?()[]{}\"'").lower()
        if not word:
            continue
        start, stop = match.span(1)
        start += body_start
        stop += body_start
        token_positions = [
            index
            for index, (left, right) in enumerate(offsets)
            if right > start and left < stop
        ]
        if not token_positions:
            continue
        by_block[token_positions[0] // block_size][word] += 1
        total[word] += 1
        for position in token_positions:
            source_positions.add(position)
            token_words[position].add(word)
            if word in answer_set:
                target_positions.add(position)
    return by_block, total, token_words, target_positions, source_positions


def exact_scores(cache, layer_idx: int, query_states: torch.Tensor, first: int, last: int):
    query = query_states.view(
        cache.batch_size,
        cache.num_key_value_heads,
        cache.num_key_value_groups,
        cache.incoming_q_len,
        cache.head_dim,
    ).float()
    if cache.query_group_mean:
        query = query.mean(dim=2, keepdim=True)
    total_tokens = cache.block_state[layer_idx].total_tokens
    tokens = cache.k_cache[
        layer_idx,
        :,
        :,
        :total_tokens,
    ].float()
    full_token_logits = torch.einsum("bhgqd,bhtd->bhgqt", query, tokens)
    full_token_logits = full_token_logits / math.sqrt(cache.head_dim)
    full_probability = torch.softmax(full_token_logits, dim=-1)
    token_logits = full_token_logits[
        ..., first * cache.block_size : last * cache.block_size
    ].view(
        cache.batch_size,
        cache.num_key_value_heads,
        query.shape[2],
        cache.incoming_q_len,
        last - first,
        cache.block_size,
    )
    block_logits = torch.logsumexp(token_logits, dim=-1)
    probability = torch.softmax(block_logits, dim=-1)
    score = probability.sum(dim=-2)
    if query.shape[2] > 1:
        score = score.amax(dim=2) if cache.group_reduce == "max" else score.sum(dim=2)
    else:
        score = score.squeeze(2)
    return score, block_logits, probability, token_logits, full_probability


def pairwise_topk_overlap(indices: torch.Tensor) -> float:
    """Mean pairwise overlap among GQA query heads, normalized by k."""
    flat = indices.reshape(-1, indices.shape[-1])
    if flat.shape[0] < 2:
        return 1.0
    sets = [set(map(int, row.tolist())) for row in flat]
    values = [
        len(sets[left] & sets[right]) / indices.shape[-1]
        for left in range(len(sets))
        for right in range(left + 1, len(sets))
    ]
    return sum(values) / len(values)


def proxy_block_logits(
    cache,
    layer_idx: int,
    query_states: torch.Tensor,
    first: int,
    last: int,
    *,
    include_alpha: bool = True,
):
    if cache.compact_metadata:
        if not include_alpha and cache.center_dispersion_correction:
            raise RuntimeError(
                "no-alpha audit for compact metadata is not implemented"
            )
        # Reuse the production packed/quantized path.  The released
        # query-mean preset has alpha disabled, so include_alpha=False is
        # identical and does not require mutating the packed metadata.
        return cache._block_logits(
            layer_idx, query_states, first, last
        )[1]
    query = query_states.view(
        cache.batch_size,
        cache.num_key_value_heads,
        cache.num_key_value_groups,
        cache.incoming_q_len,
        cache.head_dim,
    )
    centers = cache.router_centroids[layer_idx, :, :, first:last]
    logits = torch.einsum("bhgqd,bhcrd->bhgqcr", query, centers).float()
    logits = logits / math.sqrt(cache.head_dim)
    logits = logits + cache.router_log_counts[layer_idx, :, :, first:last].float()[
        :, :, None, None
    ]
    if include_alpha:
        query_norm2 = query.float().square().sum(-1) / cache.head_dim
        logits = logits + query_norm2[..., None, None] * cache.router_alpha[
            layer_idx, :, :, first:last
        ][:, :, None, None]
    return torch.logsumexp(logits, dim=-1)


def normalized_scores(cache, block_logits: torch.Tensor) -> torch.Tensor:
    score = torch.softmax(block_logits, dim=-1).sum(dim=-2)
    if cache.num_key_value_groups > 1:
        score = score.amax(dim=2) if cache.group_reduce == "max" else score.sum(dim=2)
    else:
        score = score.squeeze(2)
    return score


def selected_mass(probability: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # probability [B,H,G,Q,N], indices [B,H,K] -> [B,H]
    gather_index = indices[:, :, None, None].expand(
        -1, -1, probability.shape[2], probability.shape[3], -1
    )
    return probability.gather(-1, gather_index).sum(-1).mean(dim=(-1, -2))


def retained_word_stats(
    selected: set[int],
    fixed: set[int],
    by_block: dict[int, Counter],
    total: Counter,
    answers: list[str],
) -> tuple[float, float, int]:
    retained = Counter()
    for block in selected | fixed:
        retained.update(by_block.get(block, {}))
    ratios = [retained[word] / max(1, total[word]) for word in answers]
    top10 = {word for word, _ in retained.most_common(10)}
    return sum(ratios) / len(ratios), len(top10 & set(answers)) / len(answers), sum(retained.values())


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--gen-len", type=int, default=120)
    parser.add_argument("--detail-steps", type=int, default=4)
    parser.add_argument(
        "--partition-details", type=int, default=12,
        help="number of unique high-impact missed blocks to decode in detail",
    )
    parser.add_argument("--split-fraction", type=float, default=0.25)
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument("--candidate-factor", type=float, default=2.0)
    parser.add_argument(
        "--query-group-mean", action="store_true",
        help="average GQA sibling queries before routing, as in the querymean method",
    )
    parser.add_argument(
        "--route-mode",
        choices=(
            "deployed",
            "proxy_no_alpha",
            "proxy_raw",
            "exact_normalized",
            "exact_raw",
        ),
        default="deployed",
        help="routing score used for generation; deployed reproduces the campaign",
    )
    parser.add_argument(
        "--exact-layers",
        default="",
        help="optional layers whose deployed selection is replaced by exact block-LSE",
    )
    parser.add_argument(
        "--no-alpha-layers",
        default="",
        help="optional layers using the same centroids without the isotropic correction",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from data.metrics import multi_words
    from models import choose_model_class
    from models.centroid_router_cache import (
        evaluate_self_lse_cvar_path,
        fit_agglomerative_self_lse_paths,
    )

    row = load_row(args.dataset, args.sample_index)
    llm_class = choose_model_class(args.model)
    llm = llm_class(
        model_name=args.model,
        batch_size=1,
        device="cuda:0",
        max_length=32768 + args.gen_len + 2048,
        attn_mode=(
            "adaptive_centroid_lse_streaming_prefix4_querymean"
            if args.query_group_mean
            else "adaptive_centroid_lse_streaming_prefix4"
        ),
        dtype=torch.bfloat16,
        sparse_budget=args.budget,
        rank=160,
        chunk_size=args.block_size,
        router_centroids=args.block_size,
        router_centroid_method="self_lse_adaptive_iso",
        router_split_fraction=args.split_fraction,
        self_lse_temperatures=(1.0,),
        quest_recent_tokens=32,
        group_reduce="max",
        streaming_max_components=args.max_components,
        streaming_compact_metadata=args.query_group_mean,
        streaming_center_bits=8 if args.query_group_mean else 16,
    )
    exact_layers = parse_int_set(args.exact_layers, llm.num_layers)
    no_alpha_layers = parse_int_set(args.no_alpha_layers, llm.num_layers)
    encoded = llm.tokenizer(
        row["input"], add_special_tokens=False, return_offsets_mapping=True
    )
    input_ids = torch.tensor(encoded.input_ids, device="cuda:0", dtype=torch.long)[None]
    answers = [str(word).lower() for word in row["outputs"]]
    (
        by_block,
        source_total,
        token_words,
        target_positions,
        source_positions,
    ) = source_occurrences(
        row["input"], encoded.offset_mapping, args.block_size, answers
    )
    cache = llm.kv_cache
    original_score = cache._score_blocks
    calls_by_layer = defaultdict(int)
    call_rows: list[dict] = []
    block_rows: list[dict] = []

    def wrapped_score(layer_idx, query_states, first, last):
        step = calls_by_layer[layer_idx]
        calls_by_layer[layer_idx] += 1
        proxy_score = original_score(layer_idx, query_states, first, last)
        (
            exact_score,
            exact_logits,
            exact_probability,
            exact_token_logits,
            full_probability,
        ) = exact_scores(
            cache, layer_idx, query_states, first, last
        )
        proxy_logits = proxy_block_logits(cache, layer_idx, query_states, first, last)
        no_alpha_logits = proxy_block_logits(
            cache, layer_idx, query_states, first, last, include_alpha=False
        )
        no_alpha_score = normalized_scores(cache, no_alpha_logits)
        count = cache.select_blocks
        proxy_top = torch.topk(proxy_score, count, dim=-1).indices
        no_alpha_top = torch.topk(no_alpha_score, count, dim=-1).indices
        exact_top = torch.topk(exact_score, count, dim=-1).indices
        candidate_count = min(
            last - first, int(math.ceil(args.candidate_factor * count))
        )
        proxy_candidate = torch.topk(
            proxy_score, candidate_count, dim=-1
        ).indices
        exact_block_candidate = torch.topk(
            exact_score, candidate_count, dim=-1
        ).indices

        # Decompose loss against the exact top-B token selector.  A token is a
        # structural block miss if even exact block LSE would not shortlist its
        # block; it is a representation miss if exact block LSE would shortlist
        # the block but the small-centre proxy does not.
        dynamic_token_probability = full_probability[
            ..., first * cache.block_size:last * cache.block_size
        ]
        exact_token_score = dynamic_token_probability.mean(dim=(2, 3))
        exact_token_top = exact_token_score.topk(
            min(args.budget, exact_token_score.shape[-1]), dim=-1
        ).indices
        exact_token_blocks = exact_token_top // cache.block_size

        def token_block_covered(blocks: torch.Tensor) -> torch.Tensor:
            return (
                exact_token_blocks[..., None] == blocks[..., None, :]
            ).any(dim=-1)

        proxy_token_covered = token_block_covered(proxy_candidate)
        exact_block_token_covered = token_block_covered(exact_block_candidate)
        structural_token_miss = ~exact_block_token_covered
        representation_token_miss = (
            exact_block_token_covered & ~proxy_token_covered
        )
        raw_proxy = proxy_logits.sum(dim=-2)
        raw_exact = exact_logits.sum(dim=-2)
        if cache.num_key_value_groups > 1:
            raw_proxy = raw_proxy.amax(dim=2)
            raw_exact = raw_exact.amax(dim=2)
        else:
            raw_proxy = raw_proxy.squeeze(2)
            raw_exact = raw_exact.squeeze(2)
        raw_proxy_top = torch.topk(raw_proxy, count, dim=-1).indices
        raw_exact_top = torch.topk(raw_exact, count, dim=-1).indices
        proxy_mass = selected_mass(exact_probability, proxy_top)
        no_alpha_mass = selected_mass(exact_probability, no_alpha_top)
        exact_mass = selected_mass(exact_probability, exact_top)
        raw_proxy_mass = selected_mass(exact_probability, raw_proxy_top)
        raw_exact_mass = selected_mass(exact_probability, raw_exact_top)
        component_count = cache.component_count[layer_idx, :, :, first:last].long()
        state = cache.block_state[layer_idx]
        fixed_positions = {
            token
            for start, stop in state.exact_ranges
            for token in range(start, stop)
        }
        fixed = {
            token // cache.block_size
            for token in fixed_positions
        }
        signed_error = (proxy_logits - exact_logits).mean(dim=(-2, -3))[0]

        def block_mask(indices: torch.Tensor) -> torch.Tensor:
            mask = torch.zeros_like(proxy_score, dtype=torch.bool)
            return mask.scatter_(-1, indices, True)

        proxy_candidate_mask = block_mask(proxy_candidate)
        exact_candidate_mask = block_mask(exact_block_candidate)
        representation_missed_blocks = exact_candidate_mask & ~proxy_candidate_mask
        false_positive_blocks = proxy_candidate_mask & ~exact_candidate_mask

        total_tokens = state.total_tokens
        target_mask = torch.zeros(
            total_tokens, device=query_states.device, dtype=torch.bool
        )
        source_mask = torch.zeros_like(target_mask)
        target_ids = [position for position in target_positions if position < total_tokens]
        source_ids = [position for position in source_positions if position < total_tokens]
        if target_ids:
            target_mask[target_ids] = True
        if source_ids:
            source_mask[source_ids] = True
        distractor_mask = source_mask & ~target_mask

        flat_candidate_probability = torch.softmax(
            exact_token_logits.flatten(-2), dim=-1
        ).view_as(exact_token_logits)
        within_block_peak = (
            flat_candidate_probability.amax(dim=-1)
            / flat_candidate_probability.sum(dim=-1).clamp_min(1e-30)
        )
        per_query_top = torch.topk(
            exact_probability, count, dim=-1
        ).indices

        for head in range(cache.num_key_value_heads):
            p = proxy_top[0, head]
            na = no_alpha_top[0, head]
            e = exact_top[0, head]
            rp = raw_proxy_top[0, head]
            re = raw_exact_top[0, head]
            pset = {int(value) + first for value in p.tolist()}
            naset = {int(value) + first for value in na.tolist()}
            eset = {int(value) + first for value in e.tolist()}
            rpset = {int(value) + first for value in rp.tolist()}
            reset = {int(value) + first for value in re.tolist()}
            overlap = len(pset & eset) / count
            missed_local = sorted({int(value) for value in e.tolist()} - {int(value) for value in p.tolist()})
            exact_word_fraction, exact_top10_recall, exact_items = retained_word_stats(
                eset, fixed, by_block, source_total, answers
            )
            proxy_word_fraction, proxy_top10_recall, proxy_items = retained_word_stats(
                pset, fixed, by_block, source_total, answers
            )
            all_two = (component_count[0, head] == 2).float().mean().item()
            exact_two = (component_count[0, head, e] == 2).float().mean().item()
            missed_two = (
                (component_count[0, head, missed_local] == 2).float().mean().item()
                if missed_local else float("nan")
            )
            proxy_position_mask = torch.zeros_like(target_mask)
            exact_position_mask = torch.zeros_like(target_mask)
            if fixed_positions:
                fixed_tensor = torch.tensor(
                    sorted(fixed_positions), device=query_states.device
                )
                proxy_position_mask[fixed_tensor] = True
                exact_position_mask[fixed_tensor] = True
            for local in p.tolist():
                begin = (int(local) + first) * cache.block_size
                proxy_position_mask[begin : begin + cache.block_size] = True
            for local in e.tolist():
                begin = (int(local) + first) * cache.block_size
                exact_position_mask[begin : begin + cache.block_size] = True

            head_probability = full_probability[0, head]
            full_target_mass = head_probability[..., target_mask].sum(-1)
            full_distractor_mass = head_probability[..., distractor_mask].sum(-1)
            full_proxy_mass = head_probability[..., proxy_position_mask].sum(-1)
            full_exact_mass = head_probability[..., exact_position_mask].sum(-1)
            proxy_target_mass = head_probability[
                ..., proxy_position_mask & target_mask
            ].sum(-1)
            exact_target_mass = head_probability[
                ..., exact_position_mask & target_mask
            ].sum(-1)
            flat_probability = head_probability.flatten(-2)
            top1_mass = flat_probability.amax(-1)
            top8_mass = torch.topk(
                flat_probability, min(8, total_tokens), dim=-1
            ).values.sum(-1)
            entropy = -(
                flat_probability
                * flat_probability.clamp_min(1e-30).log()
            ).sum(-1)
            peak_positions = flat_probability.argmax(-1)
            peak_is_target = target_mask[peak_positions]
            target_count = max(1, int(target_mask.sum()))
            distractor_count = max(1, int(distractor_mask.sum()))
            target_density = full_target_mass / target_count
            distractor_density = full_distractor_mass / distractor_count
            exact_top_peak = within_block_peak[0, head].gather(
                -1,
                e[None, None].expand(
                    within_block_peak.shape[2],
                    within_block_peak.shape[3],
                    -1,
                ),
            )
            call_rows.append(
                {
                    "sample": args.sample_index,
                    "budget": args.budget,
                    "step": step,
                    "layer": layer_idx,
                    "kv_head": head,
                    "selection_overlap": overlap,
                    "no_alpha_vs_exact_normalized_overlap": len(naset & eset) / count,
                    "deployed_vs_exact_raw_overlap": len(pset & reset) / count,
                    "proxy_raw_vs_exact_raw_overlap": len(rpset & reset) / count,
                    "exact_normalized_vs_exact_raw_overlap": len(eset & reset) / count,
                    "proxy_exact_mass": proxy_mass[0, head].item(),
                    "no_alpha_exact_mass": no_alpha_mass[0, head].item(),
                    "oracle_exact_mass": exact_mass[0, head].item(),
                    "mass_gap": (exact_mass - proxy_mass)[0, head].item(),
                    "proxy_raw_exact_mass": raw_proxy_mass[0, head].item(),
                    "exact_raw_exact_mass": raw_exact_mass[0, head].item(),
                    "mean_signed_lse_error": signed_error[head].mean().item(),
                    "two_centroid_rate_all": all_two,
                    "two_centroid_rate_exact_top": exact_two,
                    "two_centroid_rate_missed_exact_top": missed_two,
                    "exact_gt_occurrence_fraction": exact_word_fraction,
                    "proxy_gt_occurrence_fraction": proxy_word_fraction,
                    "gt_occurrence_fraction_gap": exact_word_fraction - proxy_word_fraction,
                    "exact_retained_frequency_top10_recall": exact_top10_recall,
                    "proxy_retained_frequency_top10_recall": proxy_top10_recall,
                    "exact_retained_source_items": exact_items,
                    "proxy_retained_source_items": proxy_items,
                    "full_target_mass": full_target_mass.mean().item(),
                    "full_distractor_word_mass": full_distractor_mass.mean().item(),
                    "target_vs_distractor_density_ratio": (
                        target_density / distractor_density.clamp_min(1e-30)
                    ).mean().item(),
                    "full_top1_token_mass": top1_mass.mean().item(),
                    "full_top8_token_mass": top8_mass.mean().item(),
                    "full_effective_attention_tokens": entropy.exp().mean().item(),
                    "full_peak_token_is_target_rate": peak_is_target.float().mean().item(),
                    "proxy_full_attention_mass_retained": full_proxy_mass.mean().item(),
                    "exact_top_full_attention_mass_retained": full_exact_mass.mean().item(),
                    "proxy_target_attention_mass_retained": (
                        proxy_target_mass / full_target_mass.clamp_min(1e-30)
                    ).mean().item(),
                    "exact_top_target_attention_mass_retained": (
                        exact_target_mass / full_target_mass.clamp_min(1e-30)
                    ).mean().item(),
                    "exact_top_within_block_peak_share": exact_top_peak.mean().item(),
                    "gqa_query_head_topk_overlap": pairwise_topk_overlap(
                        per_query_top[0, head]
                    ),
                    "candidate_blocks": candidate_count,
                    "proxy_candidate_exact_top_token_recall": (
                        proxy_token_covered[0, head].float().mean().item()
                    ),
                    "exact_block_candidate_exact_top_token_recall": (
                        exact_block_token_covered[0, head].float().mean().item()
                    ),
                    "structural_token_miss_rate": (
                        structural_token_miss[0, head].float().mean().item()
                    ),
                    "representation_token_miss_rate": (
                        representation_token_miss[0, head].float().mean().item()
                    ),
                    "representation_missed_blocks": int(
                        representation_missed_blocks[0, head].sum().item()
                    ),
                    "false_positive_blocks": int(
                        false_positive_blocks[0, head].sum().item()
                    ),
                    "representation_miss_signed_lse_error": (
                        signed_error[head][representation_missed_blocks[0, head]]
                        .mean().item()
                        if representation_missed_blocks[0, head].any()
                        else float("nan")
                    ),
                    "false_positive_signed_lse_error": (
                        signed_error[head][false_positive_blocks[0, head]]
                        .mean().item()
                        if false_positive_blocks[0, head].any()
                        else float("nan")
                    ),
                    "representation_miss_extra_center_rate": (
                        (component_count[0, head][representation_missed_blocks[0, head]] > 1)
                        .float().mean().item()
                        if representation_missed_blocks[0, head].any()
                        else float("nan")
                    ),
                    "false_positive_extra_center_rate": (
                        (component_count[0, head][false_positive_blocks[0, head]] > 1)
                        .float().mean().item()
                        if false_positive_blocks[0, head].any()
                        else float("nan")
                    ),
                    "mean_components_all_blocks": component_count[
                        0, head
                    ].float().mean().item(),
                    "fraction_blocks_r_ge_3": (
                        component_count[0, head] >= 3
                    ).float().mean().item(),
                    "representation_miss_mean_components": (
                        component_count[0, head][representation_missed_blocks[0, head]]
                        .float().mean().item()
                        if representation_missed_blocks[0, head].any()
                        else float("nan")
                    ),
                    "false_positive_mean_components": (
                        component_count[0, head][false_positive_blocks[0, head]]
                        .float().mean().item()
                        if false_positive_blocks[0, head].any()
                        else float("nan")
                    ),
                    "exact_top_token_unique_blocks": int(
                        torch.unique(exact_token_blocks[0, head]).numel()
                    ),
                    "exact_top_tokens_per_occupied_block": (
                        exact_token_blocks.shape[-1]
                        / torch.unique(exact_token_blocks[0, head]).numel()
                    ),
                }
            )

            if step < args.detail_steps:
                exact_rank = torch.argsort(torch.argsort(exact_score[0, head], descending=True)) + 1
                proxy_rank = torch.argsort(torch.argsort(proxy_score[0, head], descending=True)) + 1
                for local in missed_local:
                    block = local + first
                    block_rows.append(
                        {
                            "sample": args.sample_index,
                            "budget": args.budget,
                            "step": step,
                            "layer": layer_idx,
                            "kv_head": head,
                            "block": block,
                            "exact_rank": int(exact_rank[local]),
                            "proxy_rank": int(proxy_rank[local]),
                            "rank_drop": int(proxy_rank[local] - exact_rank[local]),
                            "components": int(component_count[0, head, local]),
                            "signed_lse_error": float(signed_error[head, local]),
                            "source_words": " ".join(by_block.get(block, Counter()).elements()),
                            "ground_truth_words": " ".join(
                                word for word in by_block.get(block, Counter()) if word in answers
                            ),
                            "exact_block_probability": float(
                                exact_probability[0, head, ..., local].mean()
                            ),
                            "within_block_peak_share": float(
                                within_block_peak[0, head, ..., local].mean()
                            ),
                            "within_block_token_mass": json.dumps(
                                torch.softmax(
                                    exact_token_logits[
                                        0, head, ..., local, :
                                    ].float(),
                                    dim=-1,
                                ).mean(dim=(0, 1)).cpu().tolist()
                            ),
                        }
                    )
        if layer_idx in exact_layers:
            return exact_score
        if layer_idx in no_alpha_layers:
            return no_alpha_score
        return {
            "deployed": proxy_score,
            "proxy_no_alpha": no_alpha_score,
            "proxy_raw": raw_proxy,
            "exact_normalized": exact_score,
            "exact_raw": raw_exact,
        }[args.route_mode]

    cache._score_blocks = wrapped_score
    prediction = llm.generate(
        input_ids, gen_len=args.gen_len, temperature=0.0, top_p=1.0, top_k=50
    )[0]
    cache._score_blocks = original_score
    score = multi_words(prediction, row["outputs"])

    args.output.mkdir(parents=True, exist_ok=True)
    calls = pd.DataFrame(call_rows)
    blocks = pd.DataFrame(block_rows)
    calls.to_csv(args.output / "router_calls.csv", index=False)
    blocks.to_csv(args.output / "missed_exact_blocks.csv", index=False)

    # Decode a compact set of the most consequential representation misses.
    # The chosen partition is reconstructed from the exact post-RoPE keys
    # using the same robust-trimmed placement objective as the campaign.
    partition_details = []
    if not blocks.empty and args.partition_details > 0:
        ranked = blocks.copy()
        ranked["has_ground_truth"] = ranked.ground_truth_words.fillna("").ne("")
        ranked = ranked.sort_values(
            ["has_ground_truth", "exact_block_probability", "rank_drop"],
            ascending=[False, False, False],
        ).drop_duplicates(["layer", "kv_head", "block"])
        for record in ranked.head(args.partition_details).to_dict("records"):
            layer = int(record["layer"])
            head = int(record["kv_head"])
            block = int(record["block"])
            count = int(record["components"])
            begin = block * args.block_size
            stop = begin + args.block_size
            key = cache.k_cache[layer, 0, head, begin:stop].float()
            if key.shape[0] != args.block_size:
                continue
            risk_path, assignment_path = fit_agglomerative_self_lse_paths(
                key[None],
                temperatures=(1.0,),
                cost_mode="trimmed_gap",
                cost_beta=0.25,
            )
            allocation_risk = evaluate_self_lse_cvar_path(
                key[None],
                assignment_path,
                temperatures=(1.0,),
                tail_fraction=0.25,
            )
            allocation_gain = (
                allocation_risk[..., :-1] - allocation_risk[..., 1:]
            ).clamp_min(0)
            allocation_gain = torch.cummin(
                allocation_gain, dim=-1
            ).values
            allocation_threshold = float(
                cache.allocation_threshold[layer, 0, head].item()
            )
            labels = assignment_path[0, count - 1].long()
            unit = torch.nn.functional.normalize(key, dim=-1, eps=1e-12)
            cosine = unit @ unit.T
            token_ids = input_ids[0, begin:stop].detach().cpu().tolist()
            token_text = [
                llm.tokenizer.decode([token], skip_special_tokens=False)
                for token in token_ids
            ]
            clusters = []
            for label in labels.unique(sorted=True).tolist():
                members = (labels == label).nonzero(as_tuple=False).flatten()
                center = key.index_select(0, members).mean(0)
                center_unit = torch.nn.functional.normalize(
                    center, dim=0, eps=1e-12
                )
                residual = 1.0 - (
                    unit.index_select(0, members) @ center_unit
                ).clamp(-1, 1)
                clusters.append({
                    "cluster": int(label),
                    "members": members.cpu().tolist(),
                    "size": int(members.numel()),
                    "max_angular_residual": float(residual.max().item()),
                })
            partition_details.append({
                **record,
                "token_ids": token_ids,
                "token_text": token_text,
                "token_key_norm": key.norm(dim=-1).cpu().tolist(),
                "pairwise_cosine": cosine.cpu().tolist(),
                "risk_path_r1_to_r8": risk_path[0].cpu().tolist(),
                "allocation_risk_r1_to_r8": (
                    allocation_risk[0].cpu().tolist()
                ),
                "allocation_gain_r1_to_r7": (
                    allocation_gain[0].cpu().tolist()
                ),
                "allocation_threshold": allocation_threshold,
                "upgrades_above_threshold": int(
                    (allocation_gain[0] >= allocation_threshold).sum().item()
                ),
                "labels_at_allocated_r": labels.cpu().tolist(),
                "labels_path_r1_to_r8": assignment_path[0].cpu().tolist(),
                "clusters": clusters,
            })
    (args.output / "partition_details.json").write_text(
        json.dumps(partition_details, indent=2)
    )
    by_layer = calls.groupby("layer", as_index=False).mean(numeric_only=True)
    by_layer.to_csv(args.output / "by_layer.csv", index=False)
    by_head = calls.groupby("kv_head", as_index=False).mean(numeric_only=True)
    by_head.to_csv(args.output / "by_kv_head.csv", index=False)
    summary = {
        "sample": args.sample_index,
        "budget": args.budget,
        "block_size": args.block_size,
        "exact_layers": sorted(exact_layers),
        "no_alpha_layers": sorted(no_alpha_layers),
        "route_mode": args.route_mode,
        "prompt_tokens": int(input_ids.shape[-1]),
        "generated_prediction": prediction,
        "ground_truth": row["outputs"],
        "score": score,
        "calls": len(calls),
        "selection_overlap": calls.selection_overlap.mean(),
        "no_alpha_vs_exact_normalized_overlap": calls.no_alpha_vs_exact_normalized_overlap.mean(),
        "deployed_vs_exact_raw_overlap": calls.deployed_vs_exact_raw_overlap.mean(),
        "proxy_raw_vs_exact_raw_overlap": calls.proxy_raw_vs_exact_raw_overlap.mean(),
        "exact_normalized_vs_exact_raw_overlap": calls.exact_normalized_vs_exact_raw_overlap.mean(),
        "proxy_exact_mass": calls.proxy_exact_mass.mean(),
        "no_alpha_exact_mass": calls.no_alpha_exact_mass.mean(),
        "oracle_exact_mass": calls.oracle_exact_mass.mean(),
        "mass_gap": calls.mass_gap.mean(),
        "proxy_raw_exact_mass": calls.proxy_raw_exact_mass.mean(),
        "exact_raw_exact_mass": calls.exact_raw_exact_mass.mean(),
        "two_centroid_rate_all": calls.two_centroid_rate_all.mean(),
        "two_centroid_rate_exact_top": calls.two_centroid_rate_exact_top.mean(),
        "two_centroid_rate_missed_exact_top": calls.two_centroid_rate_missed_exact_top.mean(),
        "gt_occurrence_fraction_gap": calls.gt_occurrence_fraction_gap.mean(),
        "proxy_retained_frequency_top10_recall": calls.proxy_retained_frequency_top10_recall.mean(),
        "exact_retained_frequency_top10_recall": calls.exact_retained_frequency_top10_recall.mean(),
        "full_target_mass": calls.full_target_mass.mean(),
        "full_distractor_word_mass": calls.full_distractor_word_mass.mean(),
        "target_vs_distractor_density_ratio": calls.target_vs_distractor_density_ratio.mean(),
        "full_top1_token_mass": calls.full_top1_token_mass.mean(),
        "full_top8_token_mass": calls.full_top8_token_mass.mean(),
        "full_effective_attention_tokens": calls.full_effective_attention_tokens.mean(),
        "full_peak_token_is_target_rate": calls.full_peak_token_is_target_rate.mean(),
        "proxy_full_attention_mass_retained": calls.proxy_full_attention_mass_retained.mean(),
        "exact_top_full_attention_mass_retained": calls.exact_top_full_attention_mass_retained.mean(),
        "proxy_target_attention_mass_retained": calls.proxy_target_attention_mass_retained.mean(),
        "exact_top_target_attention_mass_retained": calls.exact_top_target_attention_mass_retained.mean(),
        "exact_top_within_block_peak_share": calls.exact_top_within_block_peak_share.mean(),
        "gqa_query_head_topk_overlap": calls.gqa_query_head_topk_overlap.mean(),
        "candidate_factor": args.candidate_factor,
        "split_fraction": args.split_fraction,
        "max_components": args.max_components,
        "query_group_mean": args.query_group_mean,
        "proxy_candidate_exact_top_token_recall": calls.proxy_candidate_exact_top_token_recall.mean(),
        "exact_block_candidate_exact_top_token_recall": calls.exact_block_candidate_exact_top_token_recall.mean(),
        "structural_token_miss_rate": calls.structural_token_miss_rate.mean(),
        "representation_token_miss_rate": calls.representation_token_miss_rate.mean(),
        "representation_miss_signed_lse_error": calls.representation_miss_signed_lse_error.mean(),
        "false_positive_signed_lse_error": calls.false_positive_signed_lse_error.mean(),
        "representation_miss_extra_center_rate": calls.representation_miss_extra_center_rate.mean(),
        "false_positive_extra_center_rate": calls.false_positive_extra_center_rate.mean(),
        "mean_components_all_blocks": calls.mean_components_all_blocks.mean(),
        "fraction_blocks_r_ge_3": calls.fraction_blocks_r_ge_3.mean(),
        "representation_miss_mean_components": calls.representation_miss_mean_components.mean(),
        "false_positive_mean_components": calls.false_positive_mean_components.mean(),
        "exact_top_token_unique_blocks": calls.exact_top_token_unique_blocks.mean(),
        "exact_top_tokens_per_occupied_block": calls.exact_top_tokens_per_occupied_block.mean(),
        "source_top20": source_total.most_common(20),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("worst layers by exact-mass gap")
    print(
        by_layer.sort_values("mass_gap", ascending=False)[
            [
                "layer",
                "selection_overlap",
                "mass_gap",
                "proxy_target_attention_mass_retained",
                "full_target_mass",
                "full_top8_token_mass",
                "gqa_query_head_topk_overlap",
                "two_centroid_rate_missed_exact_top",
            ]
        ].head(12).to_string(index=False)
    )
    print("worst KV heads by exact-mass gap")
    print(
        by_head.sort_values("mass_gap", ascending=False)[
            ["kv_head", "selection_overlap", "mass_gap", "two_centroid_rate_missed_exact_top"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
