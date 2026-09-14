#!/usr/bin/env python3
"""Causally diagnose missing evidence blocks in centroid-LSE routing.

For one RULER NIAH example, run an unmodified baseline followed by fixed-budget
interventions that force the mapping-line, key, or value blocks into every
selected layer/head.  Forced blocks replace the lowest-ranked selected blocks;
the sparse budget therefore remains unchanged.  The baseline also records the
rank and selection rate of every mapping-line block per layer and KV head.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
SHADOW = REPO / "ShadowKV"
sys.path.insert(0, str(SHADOW))
sys.path.insert(0, str(REPO))

from data.metrics import needle_score  # noqa: E402
from models import choose_model_class  # noqa: E402
from models.centroid_router_cache import (  # noqa: E402
    fit_exact_key_only_r_centroids,
    fit_key_only_two_centroids,
    fit_minimax_two_centroids,
    fit_self_lse_r_centroids,
    fit_self_lse_two_centroids,
)


def parse_set(spec: str, upper: int) -> set[int]:
    if spec == "all":
        return set(range(upper))
    result: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            start, stop = map(int, part.split("-", 1))
            result.update(range(start, stop + 1))
        elif part:
            result.add(int(part))
    if not result or min(result) < 0 or max(result) >= upper:
        raise ValueError(f"invalid subset {spec!r} for upper bound {upper}")
    return result


def blocks_overlapping(offsets, start: int, stop: int, block: int) -> list[int]:
    token_ids = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < stop
    ]
    if not token_ids:
        raise ValueError(f"no token overlaps character span [{start}, {stop})")
    return list(range(token_ids[0] // block, token_ids[-1] // block + 1))


def load_row(path: Path, index: int) -> dict:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(index)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument(
        "--method",
        choices=[
            "minimax2",
            "minimax2_sse",
            "scatter2",
            "cosine2",
            "self_lse2",
            "self_lse2_peak",
            "self_lse2_iso",
        ],
        required=True,
    )
    ap.add_argument("--split-fraction", type=float, default=0.25)
    ap.add_argument("--datalen", type=int, default=32768)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--outlier-blocks", type=int, default=48)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--heads", default="all")
    ap.add_argument("--steps", default="all")
    ap.add_argument(
        "--explicit-blocks",
        default="",
        help="comma-separated original block ids used by mode=explicit",
    )
    ap.add_argument(
        "--track-blocks",
        default="line",
        help="line, value, or comma-separated original block ids for baseline diagnostics",
    )
    ap.add_argument(
        "--force-split-blocks",
        default="",
        help="comma-separated blocks to give two centroids in the selected layers/heads",
    )
    ap.add_argument(
        "--target-r-values",
        default="",
        help="comma-separated r values; replace only --target-r-blocks by an exact r-centroid partition",
    )
    ap.add_argument(
        "--target-r-blocks",
        default="",
        help="comma-separated original block ids for the target-r causal sweep",
    )
    ap.add_argument(
        "--target-r-objective",
        choices=["auto", "minimax", "scatter", "cosine", "self_lse"],
        default="auto",
        help="globally optimized key-only partition used in the target-r sweep",
    )
    ap.add_argument(
        "--self-lse-temperatures",
        default="1,1.5,2",
        help="positive normalized-key query scales used by target-r-objective=self_lse",
    )
    ap.add_argument(
        "--modes",
        default="baseline,line,key,value",
        help="comma-separated subset of baseline,line,key,value,value_each,explicit",
    )
    ap.add_argument(
        "--probe-dump",
        type=Path,
        help="optional torch dump of compact block keys and decode queries for selected layers/heads",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    self_lse_temperatures = tuple(
        float(item) for item in args.self_lse_temperatures.split(",") if item
    )
    if not self_lse_temperatures or any(
        value <= 0 for value in self_lse_temperatures
    ):
        raise ValueError("--self-lse-temperatures must contain positive values")

    row = load_row(args.dataset, args.sample_index)
    answer = row["outputs"]
    while isinstance(answer, list):
        answer = answer[0]
    answer = str(answer)
    text = row["input"]
    answer_char = text.find(answer)
    if answer_char < 0:
        raise ValueError("answer is absent from prompt")
    line_start = text.rfind("\n", 0, answer_char) + 1
    line_stop = text.find("\n", answer_char)
    if line_stop < 0:
        line_stop = len(text)
    mapping_line = text[line_start:line_stop]
    key_marker = mapping_line.find(" for ")
    separator = mapping_line.find(" is:", key_marker + 5)
    if key_marker < 0 or separator < 0:
        raise ValueError(f"cannot parse mapping line: {mapping_line!r}")
    key_start = line_start + key_marker + len(" for ")
    key_stop = line_start + separator
    value_start = answer_char
    value_stop = answer_char + len(answer)

    llm_class = choose_model_class(args.model)
    llm = llm_class(
        model_name=args.model,
        batch_size=1,
        device="cuda:0",
        max_length=args.datalen + 2048,
        attn_mode="shadowkv_centroid_lse",
        dtype=torch.bfloat16,
        sparse_budget=args.budget,
        rank=160,
        chunk_size=args.block_size,
        router_centroids=2,
        router_centroid_method=args.method,
        router_split_fraction=args.split_fraction,
        self_lse_temperatures=self_lse_temperatures,
        shadow_outlier_chunks=args.outlier_blocks,
    )
    encoded = llm.tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    input_ids = torch.tensor(encoded.input_ids, device="cuda:0")[None]
    block_sets = {
        "line": blocks_overlapping(
            encoded.offset_mapping, line_start, line_stop, args.block_size
        ),
        "key": blocks_overlapping(
            encoded.offset_mapping, key_start, key_stop, args.block_size
        ),
        "value": blocks_overlapping(
            encoded.offset_mapping, value_start, value_stop, args.block_size
        ),
    }
    if args.track_blocks in {"line", "value"}:
        tracked_blocks = block_sets[args.track_blocks]
    else:
        tracked_blocks = [
            int(item) for item in args.track_blocks.split(",") if item
        ]
        if not tracked_blocks:
            raise ValueError("--track-blocks resolved to an empty set")
    force_layers = parse_set(args.layers, llm.num_layers)
    force_heads = parse_set(args.heads, llm.num_key_value_heads)
    force_steps = parse_set(args.steps, 128)
    force_split_blocks = {
        int(item) for item in args.force_split_blocks.split(",") if item
    }
    target_r_values = [
        int(item) for item in args.target_r_values.split(",") if item
    ]
    if any(not 1 <= value <= args.block_size for value in target_r_values):
        raise ValueError("--target-r-values must lie in [1, block-size]")
    target_r_blocks = {
        int(item) for item in args.target_r_blocks.split(",") if item
    }
    if target_r_values and not target_r_blocks:
        raise ValueError("--target-r-values requires --target-r-blocks")
    target_r_objective = args.target_r_objective
    if target_r_objective == "auto":
        target_r_objective = {
            "minimax2": "minimax",
            "minimax2_sse": "minimax",
            "scatter2": "scatter",
            "cosine2": "cosine",
            "self_lse2": "self_lse",
            "self_lse2_peak": "self_lse",
            "self_lse2_iso": "self_lse",
        }[args.method]
    cache = llm.kv_cache
    original_get = cache.get_retrieval_position_ids
    original_prefill = cache.prefill_kv_cache
    state = {
        "force": set(),
        "track": False,
        "records": [],
        "target_keys": {},
        "target_r": None,
        "target_models": {},
        "probe_blocks": {},
        "probe_ids": {},
        "probe_queries": defaultdict(list),
    }

    def wrapped_prefill(new_v_cache, layer_idx, key_states_roped, query=None):
        state["target_keys"][layer_idx] = {
            block: key_states_roped[
                0,
                :,
                block * cache.chunk_size : (block + 1) * cache.chunk_size,
            ].float()
            for block in tracked_blocks
        }
        result = original_prefill(new_v_cache, layer_idx, key_states_roped, query)
        if args.probe_dump is not None and layer_idx in force_layers:
            eligible = cache.chunks * cache.chunk_size
            all_blocks = key_states_roped[:, :, :eligible].view(
                cache.batch_size,
                cache.num_key_value_heads,
                cache.chunks,
                cache.chunk_size,
                cache.head_dim,
            )
            for head in force_heads:
                ids = cache.k_landmark_idx[layer_idx][0, head]
                compact = all_blocks[0, head].index_select(0, ids)
                state["probe_blocks"][(layer_idx, head)] = compact.cpu()
                state["probe_ids"][(layer_idx, head)] = ids.cpu()
        if (
            state["target_r"] is not None
            and state["target_r"] > 1
            and layer_idx in force_layers
        ):
            for head in force_heads:
                for block in target_r_blocks:
                    found = (
                        cache.k_landmark_idx[layer_idx][0, head] == block
                    ).nonzero(as_tuple=False)
                    if not found.numel():
                        continue
                    keys = key_states_roped[
                        0,
                        head,
                        block * cache.chunk_size : (block + 1) * cache.chunk_size,
                    ][None]
                    if target_r_objective == "self_lse":
                        fitted = fit_self_lse_r_centroids(
                            keys,
                            n_centroids=state["target_r"],
                            temperatures=self_lse_temperatures,
                            return_assignment=args.method == "self_lse2_iso",
                        )
                        if args.method == "self_lse2_iso":
                            centers, counts, objective, assignment = fitted
                            membership = torch.nn.functional.one_hot(
                                assignment,
                                num_classes=state["target_r"],
                            ).float()
                            residual2 = (
                                keys.float()[..., :, None, :]
                                - centers.float()[..., None, :, :]
                            ).square().sum(-1)
                            isotropic_alpha = (
                                (residual2 * membership).sum(-2)
                                / counts.float()
                                / (2.0 * cache.head_dim)
                            )
                        else:
                            centers, counts, objective = fitted
                            isotropic_alpha = None
                    else:
                        centers, counts, objective = fit_exact_key_only_r_centroids(
                            keys,
                            n_centroids=state["target_r"],
                            objective=target_r_objective,
                        )
                        isotropic_alpha = None
                    state["target_models"][(layer_idx, head, block)] = (
                        centers[0],
                        counts[0],
                        float(objective[0].item()),
                        None if isotropic_alpha is None else isotropic_alpha[0],
                    )
        if force_split_blocks and layer_idx in force_layers:
            for head in force_heads:
                for block in force_split_blocks:
                    found = (
                        cache.k_landmark_idx[layer_idx][0, head] == block
                    ).nonzero(as_tuple=False)
                    if not found.numel():
                        continue
                    compact = int(found[0, 0])
                    keys = key_states_roped[
                        0,
                        head,
                        block * cache.chunk_size : (block + 1) * cache.chunk_size,
                    ][None]
                    if args.method in {"cosine2", "scatter2"}:
                        centers, counts, *_ = fit_key_only_two_centroids(
                            keys,
                            objective={"cosine2": "cosine", "scatter2": "scatter"}[
                                args.method
                            ],
                        )
                    elif args.method in {
                        "self_lse2",
                        "self_lse2_peak",
                        "self_lse2_iso",
                    }:
                        fitted = fit_self_lse_two_centroids(
                            keys,
                            temperatures=self_lse_temperatures,
                            return_isotropic_alpha=args.method == "self_lse2_iso",
                        )
                        if args.method == "self_lse2_iso":
                            centers, counts, *_, isotropic_alpha = fitted
                        else:
                            centers, counts, *_ = fitted
                    else:
                        centers, counts, *_ = fit_minimax_two_centroids(keys)
                    cache.router_centroids[layer_idx][0, head, compact] = centers[0]
                    cache.router_log_counts[layer_idx][0, head, compact] = (
                        counts[0].float().log().to(cache.router_centroids[layer_idx].dtype)
                    )
                    cache.router_split_mask[layer_idx][0, head, compact] = True
                    if args.method == "self_lse2_iso":
                        cache.router_isotropic_alpha[layer_idx][
                            0, head, compact
                        ] = isotropic_alpha[0]
        return result

    def wrapped_get(layer_idx, query_states):
        position_ids = original_get(layer_idx, query_states).clone()
        selected = position_ids.view(
            cache.batch_size,
            cache.num_key_value_heads,
            cache.select_sets,
            cache.chunk_size,
        )[..., 0] // cache.chunk_size

        query = query_states.view(
            -1,
            cache.num_key_value_heads,
            cache.num_key_value_groups,
            query_states.shape[-2],
            cache.head_dim,
        )
        if args.probe_dump is not None and layer_idx in force_layers:
            for head in force_heads:
                state["probe_queries"][(layer_idx, head)].append(
                    query[0, head].detach().cpu()
                )
        if (
            state["target_r"] is not None
            and state["target_r"] > 1
            and layer_idx in force_layers
        ):
            component = torch.einsum(
                "bhgqd,bhcrd->bhgqcr",
                query,
                cache.router_centroids[layer_idx],
            ).float() / math.sqrt(cache.head_dim)
            component = component + cache.router_log_counts[layer_idx].float()[
                :, :, None, None, :, :
            ]
            if args.method == "self_lse2_iso":
                query_norm2 = query.float().square().sum(-1) / cache.head_dim
                component = component + query_norm2[..., None, None] * (
                    cache.router_isotropic_alpha[layer_idx][
                        :, :, None, None, :, :
                    ].float()
                )
            block_logits = torch.logsumexp(component, dim=-1)
            candidates = cache.k_landmark_idx[layer_idx]
            for head in force_heads:
                for target in target_r_blocks:
                    found = (candidates[0, head] == target).nonzero(as_tuple=False)
                    model = state["target_models"].get((layer_idx, head, target))
                    if not found.numel() or model is None:
                        continue
                    compact = int(found[0, 0])
                    centers, counts, _, isotropic_alpha = model
                    target_component = torch.einsum(
                        "gqd,rd->gqr", query[0, head].float(), centers.float()
                    ) / math.sqrt(cache.head_dim)
                    target_component = target_component + counts.float().log()[
                        None, None, :
                    ]
                    if isotropic_alpha is not None:
                        query_norm2 = (
                            query[0, head].float().square().sum(-1)
                            / cache.head_dim
                        )
                        target_component = target_component + (
                            query_norm2[..., None]
                            * isotropic_alpha.float()[None, None, :]
                        )
                    block_logits[0, head, :, :, compact] = torch.logsumexp(
                        target_component, dim=-1
                    )
            block_logits = block_logits.to(cache.dtype)
            score = torch.softmax(
                block_logits, dim=-1, dtype=torch.float32
            ).to(cache.dtype).sum(-2)
            if cache.num_key_value_groups > 1:
                score = score.max(dim=-2).values
            merged = torch.topk(score, k=cache.select_sets, dim=-1).indices
            selected = candidates.gather(dim=-1, index=merged)
            cache.selected_chunk_idx[layer_idx].copy_(selected)
            position_ids = (
                selected[..., None] * cache.chunk_size
                + torch.arange(cache.chunk_size, device=selected.device).view(
                    1, 1, 1, -1
                )
            ).view(cache.batch_size, cache.num_key_value_heads, -1)

        if state["track"]:
            component = torch.einsum(
                "bhgqd,bhcrd->bhgqcr",
                query,
                cache.router_centroids[layer_idx],
            ).float() / math.sqrt(cache.head_dim)
            component = component + cache.router_log_counts[layer_idx].float()[
                :, :, None, None, :, :
            ]
            if args.method == "self_lse2_iso":
                query_norm2 = query.float().square().sum(-1) / cache.head_dim
                component = component + query_norm2[..., None, None] * (
                    cache.router_isotropic_alpha[layer_idx][
                        :, :, None, None, :, :
                    ].float()
                )
            score = torch.logsumexp(component, dim=-1)
            block_logits = score
            score = torch.softmax(score, dim=-1, dtype=torch.float32).sum(-2)
            if cache.num_key_value_groups > 1:
                score = score.max(dim=-2).values
            candidates = cache.k_landmark_idx[layer_idx]
            for head in range(cache.num_key_value_heads):
                ids = candidates[0, head]
                for target in tracked_blocks:
                    found = (ids == target).nonzero(as_tuple=False)
                    if found.numel():
                        compact = int(found[0, 0])
                        target_score = score[0, head, compact]
                        rank = int((score[0, head] > target_score).sum().item()) + 1
                        selected_now = bool((selected[0, head] == target).any())
                        residency = "selected" if selected_now else "candidate"
                        gain = cache.router_allocation_gain[layer_idx]
                        if gain is None:
                            allocation_gain = None
                            allocation_rank = None
                        else:
                            allocation_gain = float(gain[0, head, compact].item())
                            allocation_rank = int(
                                (gain[0, head] > gain[0, head, compact]).sum().item()
                            ) + 1
                        split = bool(cache.router_split_mask[layer_idx][0, head, compact])
                        token_logits = torch.einsum(
                            "gqd,sd->gqs",
                            query[0, head].float() / math.sqrt(cache.head_dim),
                            state["target_keys"][layer_idx][target][head],
                        )
                        exact = torch.logsumexp(token_logits, dim=-1)
                        approximate = block_logits[0, head, :, :, compact]
                        corrected_logits = block_logits[0, head].clone()
                        corrected_logits[..., compact] = exact
                        corrected_score = torch.softmax(
                            corrected_logits, dim=-1, dtype=torch.float32
                        ).sum(-2)
                        if cache.num_key_value_groups > 1:
                            corrected_score = corrected_score.max(dim=-2).values
                        exact_corrected_rank = int(
                            (
                                corrected_score
                                > corrected_score[compact]
                            ).sum().item()
                        ) + 1
                        gap = exact - approximate
                        lse_gap_max = float(gap.max().item())
                        worst = int(gap.flatten().argmax().item())
                        worst_group = worst // gap.shape[-1]
                        worst_query = worst % gap.shape[-1]
                        logits_at_worst = token_logits[worst_group, worst_query]
                        peak_token_offset = int(logits_at_worst.argmax().item())
                        peak_share = float(
                            torch.softmax(logits_at_worst, dim=-1).max().item()
                        )
                        peak_minus_mean = float(
                            (logits_at_worst.max() - logits_at_worst.mean()).item()
                        )
                    else:
                        rank = 0
                        residency = "resident_outlier_or_local"
                        allocation_gain = None
                        allocation_rank = None
                        split = None
                        exact_corrected_rank = 0
                        lse_gap_max = 0.0
                        peak_token_offset = None
                        peak_share = None
                        peak_minus_mean = None
                    state["records"].append(
                        {
                            "step": int(cache.gen_offset),
                            "layer": layer_idx,
                            "head": head,
                            "block": target,
                            "rank": rank,
                            "residency": residency,
                            "split": split,
                            "allocation_gain": allocation_gain,
                            "allocation_rank": allocation_rank,
                            "exact_corrected_rank": exact_corrected_rank,
                            "lse_gap_max": lse_gap_max,
                            "peak_token_offset": peak_token_offset,
                            "peak_share": peak_share,
                            "peak_minus_mean": peak_minus_mean,
                        }
                    )

        if (
            state["force"]
            and layer_idx in force_layers
            and int(cache.gen_offset) in force_steps
        ):
            candidates = cache.k_landmark_idx[layer_idx]
            for head in force_heads:
                replacement = cache.select_sets - 1
                for target in sorted(state["force"]):
                    if (selected[0, head] == target).any():
                        continue
                    if not (candidates[0, head] == target).any():
                        # Already present in the permanently resident outlier/local set.
                        continue
                    while replacement >= 0 and selected[0, head, replacement].item() in state["force"]:
                        replacement -= 1
                    if replacement < 0:
                        break
                    selected[0, head, replacement] = target
                    replacement -= 1
            cache.selected_chunk_idx[layer_idx].copy_(selected)
            position_ids = (
                selected[..., None] * cache.chunk_size
                + torch.arange(cache.chunk_size, device=selected.device).view(1, 1, 1, -1)
            ).view(cache.batch_size, cache.num_key_value_heads, -1)
        return position_ids

    cache.get_retrieval_position_ids = wrapped_get
    cache.prefill_kv_cache = wrapped_prefill
    results = []
    baseline_records = []
    valid_modes = {"baseline", "line", "key", "value", "value_each", "explicit"}
    run_specs: list[tuple[str, set[int], int | None]] = []
    if target_r_values:
        run_specs.extend(
            (f"target_r_{value}", set(), value) for value in target_r_values
        )
    else:
        for requested in args.modes.split(","):
            if requested not in valid_modes:
                raise ValueError(f"unknown mode {requested}")
            if requested == "value_each":
                run_specs.extend(
                    (f"value_block_{block}", {block}, None)
                    for block in block_sets["value"]
                )
            elif requested == "explicit":
                explicit = {
                    int(item) for item in args.explicit_blocks.split(",") if item
                }
                if not explicit:
                    raise ValueError("mode=explicit requires --explicit-blocks")
                run_specs.append(("explicit", explicit, None))
            else:
                forced = (
                    set() if requested == "baseline" else set(block_sets[requested])
                )
                run_specs.append((requested, forced, None))

    for mode, forced, target_r in run_specs:
        state["force"] = forced
        state["target_r"] = target_r
        state["target_models"] = {}
        state["track"] = mode == "baseline" and target_r is None
        state["records"] = []
        prediction = llm.generate(
            input_ids,
            gen_len=args.gen_len,
            verbose=False,
            top_p=1.0,
            temperature=0.0,
        )[0]
        score = needle_score(prediction, answer)
        results.append(
            {
                "mode": mode,
                "forced_blocks": sorted(state["force"]),
                "target_r": target_r,
                "target_partition_stats": [
                    {
                        "layer": layer,
                        "head": head,
                        "block": block,
                        "counts": counts.tolist(),
                        "objective": objective,
                    }
                    for (layer, head, block), (_, counts, objective, _) in sorted(
                        state["target_models"].items()
                    )
                ],
                "prediction": prediction,
                "correct": score,
            }
        )
        if mode == "baseline":
            baseline_records = state["records"]
        print(f"{mode}: correct={score:.0f} prediction={prediction!r}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "dataset": str(args.dataset),
        "sample_index": args.sample_index,
        "method": args.method,
        "split_fraction": args.split_fraction,
        "answer": answer,
        "mapping_line": mapping_line,
        "block_sets": block_sets,
        "force_layers": sorted(force_layers),
        "force_heads": sorted(force_heads),
        "force_steps": sorted(force_steps),
        "force_split_blocks": sorted(force_split_blocks),
        "target_r_blocks": sorted(target_r_blocks),
        "target_r_objective": target_r_objective,
        "self_lse_temperatures": self_lse_temperatures,
        "results": results,
        "baseline_records": baseline_records,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    if args.probe_dump is not None:
        args.probe_dump.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": args.model,
                "dataset": str(args.dataset),
                "sample_index": args.sample_index,
                "answer": answer,
                "mapping_line": mapping_line,
                "block_sets": block_sets,
                "block_size": args.block_size,
                "blocks": state["probe_blocks"],
                "block_ids": state["probe_ids"],
                "queries": {
                    key: torch.cat(value, dim=1)
                    for key, value in state["probe_queries"].items()
                },
                "results": results,
            },
            args.probe_dump,
        )
        print(f"saved probe dump {args.probe_dump}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
