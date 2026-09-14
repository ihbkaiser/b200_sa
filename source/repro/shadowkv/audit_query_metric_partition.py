#!/usr/bin/env python3
"""Does a causal query second moment repair within-block dilution?

The production self-K path, prompt-query metrics, and a future-query oracle are
evaluated on exactly the same important missed blocks.  Future decode queries
are labels only.  The deployable prompt metrics use post-RoPE queries observed
during the prompt and never inspect generated tokens or benchmark answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))


def load_row(path: Path, index: int) -> dict:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(index)


def reduce_gqa(query: torch.Tensor, kv_heads: int) -> torch.Tensor:
    batch, query_heads, tokens, dim = query.shape
    groups = query_heads // kv_heads
    return query.view(batch, kv_heads, groups, tokens, dim).float().mean(2)


@lru_cache(None)
def set_partitions(size: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """All unlabeled set partitions of range(size), grouped only later by r."""
    output: list[tuple[tuple[int, ...], ...]] = []

    def visit(index: int, groups: list[list[int]]) -> None:
        if index == size:
            output.append(tuple(tuple(group) for group in groups))
            return
        for group in groups:
            group.append(index)
            visit(index + 1, groups)
            group.pop()
        groups.append([index])
        visit(index + 1, groups)
        groups.pop()

    visit(0, [])
    return tuple(output)


@lru_cache(None)
def partitions_by_count(size: int, count: int):
    return tuple(p for p in set_partitions(size) if len(p) == count)


def subset_mask(group) -> int:
    return sum(1 << index for index in group)


def centroid_logits(key: torch.Tensor, query: torch.Tensor, partition) -> torch.Tensor:
    values = []
    scale = math.sqrt(key.shape[-1])
    for group in partition:
        center = key[list(group)].mean(0)
        values.append(query @ center / scale + math.log(len(group)))
    return torch.stack(values)


def mass_ratio(key: torch.Tensor, query: torch.Tensor, partition) -> float:
    exact = torch.logsumexp(query @ key.T / math.sqrt(key.shape[-1]), dim=-1)
    proxy = torch.logsumexp(centroid_logits(key, query, partition), dim=-1)
    return float(torch.exp(proxy - exact))


def metric_loss(key: torch.Tensor, metric: torch.Tensor, partition, mode: str) -> float:
    losses = []
    for group in partition:
        points = key[list(group)]
        residual = points - points.mean(0, keepdim=True)
        quadratic = torch.einsum("nd,de,ne->n", residual, metric, residual)
        if mode == "sum_scatter":
            losses.append(quadratic.sum())
        elif mode == "max_radius":
            losses.append(quadratic.max())
        else:
            raise ValueError(mode)
    if mode == "sum_scatter":
        return float(torch.stack(losses).sum())
    return float(torch.stack(losses).max())


def best_metric_partition(key: torch.Tensor, metric: torch.Tensor, count: int, mode: str):
    # Only 2^8-1 distinct clusters exist.  Evaluate each cluster once, then
    # combine scalar losses over the 4,140 set partitions.
    subset_loss = {}
    for mask in range(1, 1 << len(key)):
        group = [index for index in range(len(key)) if mask & (1 << index)]
        subset_loss[mask] = metric_loss(key, metric, (group,), mode)
    candidates = partitions_by_count(len(key), count)
    combine = sum if mode == "sum_scatter" else max
    return min(
        candidates,
        key=lambda p: combine(subset_loss[subset_mask(group)] for group in p),
    )


def best_oracle_partition(key: torch.Tensor, query: torch.Tensor, count: int):
    scale = math.sqrt(key.shape[-1])
    subset_mass = {}
    for mask in range(1, 1 << len(key)):
        group = [index for index in range(len(key)) if mask & (1 << index)]
        center = key[group].mean(0)
        subset_mass[mask] = float(len(group) * torch.exp(query @ center / scale))
    candidates = partitions_by_count(len(key), count)
    return max(
        candidates,
        key=lambda p: sum(subset_mass[subset_mask(group)] for group in p),
    )


def best_empirical_partition(
    key: torch.Tensor,
    query_bank: torch.Tensor,
    count: int,
    mode: str,
):
    """Minimize exact Jensen loss over a bank of causal/oracle queries."""
    scale = math.sqrt(key.shape[-1])
    exact = torch.logsumexp(query_bank @ key.T / scale, dim=-1)
    subset_mass = {}
    for mask in range(1, 1 << len(key)):
        group = [index for index in range(len(key)) if mask & (1 << index)]
        center = key[group].mean(0)
        subset_mass[mask] = len(group) * torch.exp(query_bank @ center / scale)

    def loss(partition):
        proxy = sum(subset_mass[subset_mask(group)] for group in partition)
        gap = exact - proxy.clamp_min(1e-30).log()
        if mode == "mean":
            return float(gap.mean())
        if mode == "tail25":
            return float(gap.topk(max(1, math.ceil(0.25 * len(gap)))).values.mean())
        raise ValueError(mode)

    return min(partitions_by_count(len(key), count), key=loss)


def labels_to_partition(labels: list[int]):
    unique = sorted(set(labels))
    return tuple(tuple(i for i, value in enumerate(labels) if value == label) for label in unique)


def min_count_for(path: list[float], threshold: float) -> int:
    for index, value in enumerate(path, 1):
        if value >= threshold:
            return index
    return len(path)


def top_eigen_projector(covariance: torch.Tensor, rank: int) -> torch.Tensor:
    _, vector = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    basis = vector[:, -rank:]
    return basis @ basis.T


def matrix_root(covariance: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    value, vector = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    cutoff = value.max().clamp_min(1e-30) * 1e-7
    if inverse:
        diagonal = torch.where(value > cutoff, value.rsqrt(), torch.zeros_like(value))
    else:
        diagonal = value.clamp_min(0).sqrt()
    return (vector * diagonal[None]) @ vector.T


def optimal_bilinear_operator(
    query_second_moment: torch.Tensor,
    residual_second_moment: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Best rank-r M for E[(q^T r-q^T M r)^2] under product sampling."""
    qroot = matrix_root(query_second_moment)
    rroot = matrix_root(residual_second_moment)
    cross = qroot @ rroot
    left, singular, right_h = torch.linalg.svd(cross, full_matrices=False)
    approximation = (
        left[:, :rank] * singular[:rank][None]
    ) @ right_h[:rank]
    return matrix_root(query_second_moment, inverse=True) @ approximation @ matrix_root(
        residual_second_moment, inverse=True
    )


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--center-bits", type=int, choices=(4, 8, 16), default=8)
    ap.add_argument("--gen-len", type=int, default=120)
    ap.add_argument("--prompt-samples", type=int, default=512)
    ap.add_argument("--recent-window", type=int, default=512)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from models import choose_model_class
    from models.adaptive_centroid_streaming_cache import (
        _dequantize_centers,
        _key_pca_factors,
        _optimal_bilinear_factors,
        _quantize_centers,
    )

    records = json.loads(args.records.read_text())
    wanted_layers = {int(record["layer"]) for record in records}
    row = load_row(args.dataset, args.sample_index)
    llm = choose_model_class(args.model)(
        model_name=args.model,
        batch_size=1,
        device="cuda:0",
        max_length=32768 + args.gen_len + 2048,
        attn_mode="adaptive_centroid_lse_streaming_prefix4_querymean",
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
        streaming_max_components=8,
        streaming_compact_metadata=True,
        streaming_center_bits=args.center_bits,
    )
    cache = llm.kv_cache
    prompt_metric: dict[int, dict[str, torch.Tensor]] = {}
    prompt_bank: dict[int, dict[str, torch.Tensor]] = {}
    decode_queries: dict[int, list[torch.Tensor]] = defaultdict(list)
    route_ranges: dict[tuple[int, int], tuple[int, int]] = {}
    coarse_scores: dict[tuple[int, int], torch.Tensor] = {}
    prefill_layer = 0
    decode_calls = 0
    original_rope = llm.apply_rotary_pos_emb
    original_block_logits = cache._block_logits

    def wrapped_rope(query, key, position_ids):
        nonlocal prefill_layer, decode_calls
        query, key = original_rope(query, key, position_ids)
        if query.shape[-2] > 1:
            layer = prefill_layer
            prefill_layer += 1
            if layer in wanted_layers:
                q = reduce_gqa(query.detach(), cache.num_key_value_heads)[0]
                length = q.shape[-2]
                count = min(args.prompt_samples, length)
                index = torch.linspace(0, length - 1, count, device=q.device).round().long()
                global_q = q.index_select(-2, index)
                recent_q = q[..., -min(args.recent_window, length):, :]
                prompt_metric[layer] = {}
                prompt_bank[layer] = {
                    "global": global_q.cpu(),
                    "recent": recent_q.cpu(),
                }
                for name, bank in (("global", global_q), ("recent", recent_q)):
                    # The uncentered second moment is required because the
                    # relevant quantity is E[(q^T d)^2], not Var(q^T d).
                    moment = torch.einsum("htd,hte->hde", bank, bank) / bank.shape[-2]
                    centered = bank - bank.mean(-2, keepdim=True)
                    covariance = torch.einsum("htd,hte->hde", centered, centered) / bank.shape[-2]
                    prompt_metric[layer][f"{name}_moment"] = moment.cpu()
                    prompt_metric[layer][f"{name}_covariance"] = covariance.cpu()
        else:
            layer = decode_calls % llm.num_layers
            decode_calls += 1
            if layer in wanted_layers:
                decode_queries[layer].append(
                    reduce_gqa(query.detach(), cache.num_key_value_heads)[0, :, 0].cpu()
                )
        return query, key

    def wrapped_block_logits(layer_idx, query_states, first_block, last_block):
        output = original_block_logits(
            layer_idx, query_states, first_block, last_block
        )
        if layer_idx in wanted_layers:
            step = len(decode_queries[layer_idx]) - 1
            route_ranges[(step, layer_idx)] = (
                int(first_block), int(last_block)
            )
            coarse_scores[(step, layer_idx)] = cache._reduce_block_logits(
                output[1]
            )[0].detach().cpu()
        return output

    llm.apply_rotary_pos_emb = wrapped_rope
    cache._block_logits = wrapped_block_logits
    input_ids = torch.tensor(
        llm.tokenizer.encode(row["input"], add_special_tokens=False),
        device="cuda:0", dtype=torch.long,
    )[None]
    prediction = llm.generate(
        input_ids, gen_len=args.gen_len, temperature=0.0, top_p=1.0, top_k=50
    )[0]
    llm.apply_rotary_pos_emb = original_rope
    cache._block_logits = original_block_logits

    rows = []
    empirical_rows = []
    sketch_rows = []
    ranking_rows = []
    cascade_rows = []
    residual_covariance_cache: dict[tuple[int, int], torch.Tensor] = {}
    key_second_moment_cache: dict[tuple[int, int], torch.Tensor] = {}
    operator_cache: dict[tuple[int, int, str, int], dict[str, torch.Tensor]] = {}
    direct_factor_cache: dict[
        tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]
    ] = {}
    key_pca_factor_cache: dict[
        tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]
    ] = {}
    reproduction_errors = []
    for record_index, record in enumerate(records):
        layer = int(record["layer"])
        head = int(record["kv_head"])
        block = int(record["block"])
        step = int(record["step"])
        begin, stop = block * 8, (block + 1) * 8
        key = cache.k_cache[layer, 0, head, begin:stop].float().cpu()
        if step >= len(decode_queries[layer]):
            continue
        query = decode_queries[layer][step][head].float()
        actual_mass = torch.softmax(query @ key.T / math.sqrt(key.shape[-1]), dim=-1)
        expected_value = record["within_block_token_mass"]
        if isinstance(expected_value, str):
            expected_value = json.loads(expected_value)
        expected_mass = torch.tensor(expected_value)
        reproduction_errors.append(float((actual_mass - expected_mass).abs().max()))

        metrics = {
            name: tensor[head].float()
            for name, tensor in prompt_metric[layer].items()
        }
        future_bank = torch.stack(decode_queries[layer], dim=1)[head].float()
        future_moment = torch.einsum(
            "td,te->de", future_bank, future_bank
        ) / len(future_bank)
        metric_methods = {
            "prompt_global_moment_sum": (metrics["global_moment"], "sum_scatter"),
            "prompt_global_moment_max": (metrics["global_moment"], "max_radius"),
            "prompt_recent_moment_max": (metrics["recent_moment"], "max_radius"),
            "prompt_global_cov_max": (metrics["global_covariance"], "max_radius"),
            # Oracle distribution control: this sees all future decode-query
            # directions for the head, but not which query activates this
            # block.  It distinguishes temporal distribution shift from the
            # harder need for the particular live query.
            "future_decode_moment_sum": (future_moment, "sum_scatter"),
            "future_decode_moment_max": (future_moment, "max_radius"),
        }
        paths: dict[str, list] = {
            "self_k_current": [
                labels_to_partition(labels)
                for labels in record["labels_path_r1_to_r8"]
            ]
        }
        for name, (metric, mode) in metric_methods.items():
            paths[name] = [
                best_metric_partition(key, metric, count, mode)
                for count in range(1, 9)
            ]
        paths["future_query_oracle"] = [
            best_oracle_partition(key, query, count) for count in range(1, 9)
        ]

        peak = int(actual_mass.argmax())
        for method, path in paths.items():
            ratios = [mass_ratio(key, query, partition) for partition in path]
            isolation = [
                any(len(group) == 1 and group[0] == peak for group in partition)
                for partition in path
            ]
            for count, (partition, ratio, isolated) in enumerate(
                zip(path, ratios, isolation), 1
            ):
                rows.append({
                    "record": record_index,
                    "layer": layer,
                    "kv_head": head,
                    "block": block,
                    "step": step,
                    "exact_rank": int(record["exact_rank"]),
                    "proxy_rank": int(record["proxy_rank"]),
                    "allocated_r": int(record["components"]),
                    "method": method,
                    "r": count,
                    "mass_ratio": ratio,
                    "peak_is_singleton": int(isolated),
                    "partition": "|".join(
                        ",".join(map(str, group)) for group in partition
                    ),
                    "r50": min_count_for(ratios, 0.5),
                    "r90": min_count_for(ratios, 0.9),
                })

        # At low mean center budgets almost all upgraded blocks have r=2.
        # Optimize this exact decision under whole query banks to determine
        # whether the second-moment approximation or static partitioning is
        # the limiting factor.
        empirical_banks = {
            "prompt_exact_mean": (prompt_bank[layer]["global"][head].float(), "mean"),
            "prompt_exact_tail25": (prompt_bank[layer]["global"][head].float(), "tail25"),
            "future_exact_mean": (future_bank, "mean"),
            "future_exact_tail25": (future_bank, "tail25"),
        }
        for method, (bank, mode) in empirical_banks.items():
            partition = best_empirical_partition(key, bank, 2, mode)
            empirical_rows.append({
                "record": record_index,
                "layer": layer,
                "kv_head": head,
                "block": block,
                "step": step,
                "method": method,
                "mass_ratio": mass_ratio(key, query, partition),
                "peak_is_singleton": int(any(
                    len(group) == 1 and group[0] == peak for group in partition
                )),
                "partition": "|".join(
                    ",".join(map(str, group)) for group in partition
                ),
            })

        # Shared-basis residual sketch.  The cache is built from a uniform
        # sample of prompt blocks for this layer/head; no future query or task
        # label enters deployable operators.
        cache_key = (layer, head)
        if cache_key not in residual_covariance_cache:
            usable = (input_ids.shape[-1] // 8) * 8
            n_blocks = usable // 8
            take = min(512, n_blocks)
            block_index = torch.linspace(0, n_blocks - 1, take).round().long()
            position = (
                block_index[:, None] * 8 + torch.arange(8)[None]
            ).flatten()
            sampled_key = cache.k_cache[layer, 0, head, position].float().cpu().view(
                take, 8, -1
            )
            sampled_residual = sampled_key - sampled_key.mean(1, keepdim=True)
            flat_residual = sampled_residual.flatten(0, 1)
            residual_covariance_cache[cache_key] = (
                flat_residual.T @ flat_residual / len(flat_residual)
            )
            flat_key = sampled_key.flatten(0, 1)
            key_second_moment_cache[cache_key] = (
                flat_key.T @ flat_key / len(flat_key)
            )
        residual_covariance = residual_covariance_cache[cache_key]
        key_second_moment = key_second_moment_cache[cache_key]
        residual = key - key.mean(0, keepdim=True)
        center_logit = query @ key.mean(0) / math.sqrt(key.shape[-1])
        exact_logits = query @ key.T / math.sqrt(key.shape[-1])
        exact_lse = torch.logsumexp(exact_logits, dim=-1)

        for rank in (4, 8, 16, 32, 64):
            shared_key = (layer, head, "shared", rank)
            if shared_key not in operator_cache:
                generator = torch.Generator().manual_seed(
                    1729 + 1009 * layer + 7919 * head
                )
                random_basis = torch.linalg.qr(
                    torch.randn(key.shape[-1], rank, generator=generator),
                    mode="reduced",
                ).Q
                operator_cache[shared_key] = {
                    "random_shared": random_basis @ random_basis.T,
                    "prompt_query_pca": top_eigen_projector(
                        metrics["global_moment"], rank
                    ),
                    "prompt_residual_pca": top_eigen_projector(
                        residual_covariance, rank
                    ),
                    "prompt_joint_bilinear": optimal_bilinear_operator(
                        metrics["global_moment"], residual_covariance, rank
                    ),
                    "prompt_joint_direct_k": optimal_bilinear_operator(
                        metrics["global_moment"], key_second_moment, rank
                    ),
                    "future_query_pca_oracle": top_eigen_projector(
                        future_moment, rank
                    ),
                    "future_joint_bilinear_oracle": optimal_bilinear_operator(
                        future_moment, residual_covariance, rank
                    ),
                }
                direct_factor_cache[(layer, head, rank)] = (
                    _optimal_bilinear_factors(
                        metrics["global_moment"], key_second_moment, rank
                    )
                )
                key_pca_factor_cache[(layer, head, rank)] = (
                    _key_pca_factors(key_second_moment, rank)
                )
            operators = dict(operator_cache[shared_key])
            # Per-block residual PCA is an optimistic storage oracle: it is
            # exact by rank 7, but its 128 x r basis cannot be shared.
            operators["per_block_residual_pca_oracle"] = top_eigen_projector(
                residual.T @ residual / len(residual), rank
            )
            for method, operator in operators.items():
                if method == "prompt_joint_direct_k":
                    approximate_logits = (
                        query @ operator @ key.T
                    ) / math.sqrt(key.shape[-1])
                else:
                    approximate_logits = center_logit + (
                        query @ operator @ residual.T
                    ) / math.sqrt(key.shape[-1])
                approximate_lse = torch.logsumexp(approximate_logits, dim=-1)
                sketch_rows.append({
                    "record": record_index,
                    "layer": layer,
                    "kv_head": head,
                    "block": block,
                    "step": step,
                    "method": method,
                    "rank": rank,
                    "top_token_hit": int(approximate_logits.argmax() == exact_logits.argmax()),
                    "log_lse_error": float(approximate_lse - exact_lse),
                    "abs_log_lse_error": float((approximate_lse - exact_lse).abs()),
                    "logit_rmse": float((approximate_logits - exact_logits).square().mean().sqrt()),
                    "mass_ratio": float(torch.exp(approximate_lse - exact_lse)),
                })

    # Full-context block-ranking audit on each distinct live query represented
    # by the failure cohort.  This is the relevant test for a router: accurate
    # peak identity inside a known block is not enough.
    distinct_calls = sorted({
        (int(record["step"]), int(record["layer"]), int(record["kv_head"]))
        for record in records
    })
    select_blocks = args.budget // 8
    for step, layer, head in distinct_calls:
        if (step, layer) not in route_ranges or step >= len(decode_queries[layer]):
            continue
        first, last = route_ranges[(step, layer)]
        key = cache.k_cache[
            layer, 0, head, first * 8:last * 8
        ].float().view(last - first, 8, -1)
        query = decode_queries[layer][step][head].float().to(key.device)
        center = key.mean(1)
        residual = key - center[:, None]
        scale = math.sqrt(key.shape[-1])
        exact_logits = torch.einsum("d,nsd->ns", query, key) / scale
        exact_lse = torch.logsumexp(exact_logits, dim=-1)
        exact_probability = torch.softmax(exact_lse, dim=-1)
        exact_ids = exact_lse.topk(select_blocks).indices
        exact_mask = torch.zeros(len(key), device=key.device, dtype=torch.bool)
        exact_mask[exact_ids] = True
        oracle_mass = exact_probability[exact_mask].sum()
        center_lse = query @ center.T / scale + math.log(8)
        candidates = {"center_only": center_lse}
        for rank in (8, 16, 32, 64):
            shared = operator_cache[(layer, head, "shared", rank)]
            for method in ("prompt_query_pca", "prompt_joint_bilinear"):
                operator = shared[method].to(key.device)
                projected_query = query @ operator
                token_logits = (
                    query @ center.T / scale
                )[:, None] + torch.einsum(
                    "d,nsd->ns", projected_query, residual
                ) / scale
                candidates[f"{method}_r{rank}"] = torch.logsumexp(
                    token_logits, dim=-1
                )
            direct_operator = shared["prompt_joint_direct_k"].to(key.device)
            direct_logits = torch.einsum(
                "d,nsd->ns", query @ direct_operator, key
            ) / scale
            candidates[f"prompt_joint_direct_k_r{rank}"] = torch.logsumexp(
                direct_logits, dim=-1
            )
        for method, score in candidates.items():
            for factor in (1, 2, 4):
                count = min(len(score), factor * select_blocks)
                ids = score.topk(count).indices
                mask = torch.zeros_like(exact_mask)
                mask[ids] = True
                ranking_rows.append({
                    "step": step,
                    "layer": layer,
                    "kv_head": head,
                    "method": method,
                    "candidate_factor": factor,
                    "exact_top_block_recall": float(
                        (mask & exact_mask).sum() / select_blocks
                    ),
                    "exact_mass_covered": float(exact_probability[mask].sum()),
                    "fraction_of_oracle_mass": float(
                        exact_probability[mask].sum() / oracle_mass
                    ),
                })

        # Production-shaped cascade: adaptive quantized centroids score every
        # block; only their 2B/4B shortlist reads direct-K rank-16 INT4 codes;
        # the final decision is B individual tokens. Compare the deployed
        # query-aware factorization against the K-only PCA/SVD control.
        coarse = coarse_scores[(step, layer)][head].to(key.device)
        exact_token_logits = exact_logits.flatten()
        exact_token_probability = torch.softmax(exact_token_logits, dim=-1)
        token_budget = min(args.budget, exact_token_logits.numel())
        oracle_token = exact_token_logits.topk(token_budget).indices
        oracle_token_mass = exact_token_probability[oracle_token].sum()
        offset = torch.arange(8, device=key.device)
        for basis, factors in (
            ("query_aware", direct_factor_cache[(layer, head, 16)]),
            ("key_pca", key_pca_factor_cache[(layer, head, 16)]),
        ):
            query_factor, key_factor = factors
            projected_query = query @ query_factor.to(query.device)
            projected_key = torch.einsum(
                "nsd,dr->nsr", key, key_factor.to(key.device)
            )
            encoded, quant_scale = _quantize_centers(projected_key, 4)
            projected_key = _dequantize_centers(
                encoded, quant_scale, 4, 16, torch.float32
            )
            approximate_token_logits = torch.einsum(
                "r,nsr->ns", projected_query, projected_key
            ) / scale
            for factor in (2, 4):
                candidate_blocks = min(len(key), factor * select_blocks)
                block_ids = coarse.topk(candidate_blocks).indices
                candidate_positions = (
                    block_ids[:, None] * 8 + offset[None]
                ).flatten()
                local = approximate_token_logits[block_ids].flatten().topk(
                    token_budget
                ).indices
                selected = candidate_positions[local]
                selected_mass = exact_token_probability[selected].sum()
                overlap = torch.isin(selected, oracle_token).float().mean()

                exact_local = exact_logits[block_ids].flatten().topk(
                    token_budget
                ).indices
                exact_selected = candidate_positions[exact_local]
                exact_selected_mass = exact_token_probability[exact_selected].sum()
                cascade_rows.append({
                    "step": step,
                    "layer": layer,
                    "kv_head": head,
                    "basis": basis,
                    "center_bits": args.center_bits,
                    "candidate_factor": factor,
                    "candidate_blocks": candidate_blocks,
                    "top_token_overlap": float(overlap),
                    "selected_mass": float(selected_mass),
                    "fraction_of_oracle_mass": float(
                        selected_mass / oracle_token_mass
                    ),
                    "exact_candidate_mass": float(exact_selected_mass),
                    "exact_candidate_fraction_of_oracle_mass": float(
                        exact_selected_mass / oracle_token_mass
                    ),
                })

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "empirical_r2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(empirical_rows[0]))
        writer.writeheader()
        writer.writerows(empirical_rows)
    with (args.output / "residual_sketch.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sketch_rows[0]))
        writer.writeheader()
        writer.writerows(sketch_rows)
    with (args.output / "residual_sketch_ranking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking_rows[0]))
        writer.writeheader()
        writer.writerows(ranking_rows)
    with (args.output / "rank16_int4_cascade.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cascade_rows[0]))
        writer.writeheader()
        writer.writerows(cascade_rows)

    summary = []
    for method in sorted({row["method"] for row in rows}):
        chosen = [row for row in rows if row["method"] == method]
        for count in range(1, 9):
            subset = [row for row in chosen if row["r"] == count]
            summary.append({
                "method": method,
                "r": count,
                "median_mass_ratio": float(np.median([row["mass_ratio"] for row in subset])),
                "mean_mass_ratio": float(np.mean([row["mass_ratio"] for row in subset])),
                "peak_singleton_rate": float(np.mean([row["peak_is_singleton"] for row in subset])),
                "median_r50": float(np.median([row["r50"] for row in subset])),
                "median_r90": float(np.median([row["r90"] for row in subset])),
            })
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metadata = {
        "model": args.model,
        "dataset": str(args.dataset),
        "sample": args.sample_index,
        "prediction": prediction,
        "n_records": len({row["record"] for row in rows}),
        "max_reproduction_error": max(reproduction_errors, default=None),
        "future_queries_are_labels_only": True,
        "prompt_metric": "post-RoPE GQA-mean E[q q^T]",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
