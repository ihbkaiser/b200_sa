#!/usr/bin/env python3
"""Causal retrieval audit for a failed 128K streaming example.

Run the deployed adaptive-centroid router twice at the same sparse budget:
unaltered, then with every prompt block containing a reference answer forced
into the dynamic set by replacing the lowest-ranked selected blocks.  The
second run therefore changes selection, not capacity.  The baseline decode
queries and selected block ids are saved for offline selector diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ShadowKV"))


def load_row(path: Path, index: int) -> dict:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(index)


def flatten_strings(value) -> list[str]:
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def evidence_blocks(text: str, offsets, answers: list[str], block_size: int) -> list[int]:
    """All prompt blocks containing a literal reference answer occurrence."""
    spans: list[tuple[int, int]] = []
    lower = text.lower()
    for answer in answers:
        pattern = re.compile(r"(?<!\w)" + re.escape(answer.lower()) + r"(?!\w)")
        spans.extend((match.start(), match.end()) for match in pattern.finditer(lower))
    blocks: set[int] = set()
    for left, right in spans:
        token_positions = [
            index for index, (start, stop) in enumerate(offsets)
            if stop > left and start < right
        ]
        if token_positions:
            blocks.update(
                range(token_positions[0] // block_size,
                      token_positions[-1] // block_size + 1)
            )
    if not blocks:
        raise RuntimeError("no literal answer occurrence was found in the prompt")
    return sorted(blocks)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--task", choices=("cwe", "niah_multikey_3"), required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--datalen", type=int, default=131072)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--mean-components", type=float, default=1.5)
    ap.add_argument("--gen-len", type=int, default=0)
    ap.add_argument(
        "--center-placement",
        choices=("robust_trimmed", "qmass_one", "qmass_bank"),
        default="robust_trimmed",
    )
    ap.add_argument("--query-mass-bank-size", type=int, default=4)
    ap.add_argument("--query-mass-window", type=int, default=4)
    ap.add_argument(
        "--query-mass-gqa-reduce", choices=("mean", "all"), default="mean"
    )
    ap.add_argument(
        "--previous-selection-fraction", type=float, default=0.0,
        help=(
            "diagnostic temporal reservoir: reserve this fraction of the "
            "budget for blocks selected most often in the recent raw rankings"
        ),
    )
    ap.add_argument("--selection-history-window", type=int, default=1)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument(
        "--lightweight", action="store_true",
        help="skip per-head recall records and trace copies",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 0.0 <= args.previous_selection_fraction < 1.0:
        ap.error("--previous-selection-fraction must lie in [0,1)")
    if args.selection_history_window < 1:
        ap.error("--selection-history-window must be positive")

    os.environ["SHADOWKV_CENTER_PLACEMENT"] = args.center_placement
    os.environ["SHADOWKV_CENTER_ALLOCATION"] = (
        "tail_cvar" if args.center_placement == "robust_trimmed" else "marginal"
    )
    os.environ["SHADOWKV_ROBUST_TRIM_FRACTION"] = "0.25"
    os.environ["SHADOWKV_TAIL_CVAR_FRACTION"] = "0.25"
    os.environ["SHADOWKV_TAIL_GAP_CORRECTION_SCALE"] = "0.25"
    os.environ["SHADOWKV_CENTER_DISPERSION_CORRECTION"] = "0"
    os.environ["SHADOWKV_QUERY_MASS_BANK_SIZE"] = str(args.query_mass_bank_size)
    os.environ["SHADOWKV_QUERY_MASS_WINDOW"] = str(args.query_mass_window)
    os.environ["SHADOWKV_QUERY_MASS_GQA_REDUCE"] = args.query_mass_gqa_reduce

    from data.metrics import multi_words, needle_score
    from models import choose_model_class

    row = load_row(args.dataset, args.sample_index)
    answers = flatten_strings(row["outputs"])
    llm_class = choose_model_class(args.model)
    llm = llm_class(
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
    encoded = llm.tokenizer(
        row["input"], add_special_tokens=False, return_offsets_mapping=True
    )
    input_ids = torch.tensor(encoded.input_ids, device="cuda:0")[None]
    target_blocks = evidence_blocks(
        row["input"], encoded.offset_mapping, answers, llm.kv_cache.block_size
    )
    gen_len = args.gen_len or (120 if args.task == "cwe" else 128)

    cache = llm.kv_cache
    original_select = cache._select_block_ids
    mode = "baseline"
    call_count: defaultdict[int, int] = defaultdict(int)
    selection_history: dict[int, deque[torch.Tensor]] = defaultdict(
        lambda: deque(maxlen=args.selection_history_window)
    )
    selection_frequency: dict[int, torch.Tensor] = {}
    selection_last_priority: dict[int, torch.Tensor] = {}
    rows: list[dict] = []
    traces: list[dict] = []

    def wrapped_select(layer_idx: int, query_states: torch.Tensor) -> torch.Tensor:
        raw_selected = original_select(layer_idx, query_states).clone()
        selected = raw_selected.clone()
        step = call_count[layer_idx]
        call_count[layer_idx] += 1
        first, last = cache.block_state[layer_idx].candidate_block_range
        history = selection_history[layer_idx]
        reserve = int(round(
            selected.shape[-1] * args.previous_selection_fraction
        ))
        if history and reserve:
            frequency = selection_frequency[layer_idx]
            last_priority = selection_last_priority[layer_idx]
            scale = float(
                args.selection_history_window * (selected.shape[-1] + 1) + 1
            )
            reservoir_score = (
                frequency[..., first:last].float() * scale
                + last_priority[..., first:last]
            )
            keep_all = reservoir_score.topk(reserve, dim=-1).indices + first
            duplicate = (
                raw_selected[..., :, None] == keep_all[..., None, :]
            ).any(-1)
            position = torch.arange(
                raw_selected.shape[-1], device=selected.device
            ).expand_as(raw_selected)
            fresh_index = position.masked_fill(
                duplicate, raw_selected.shape[-1]
            ).topk(
                raw_selected.shape[-1] - reserve,
                dim=-1, largest=False, sorted=True,
            ).indices
            fresh = raw_selected.gather(-1, fresh_index)
            selected = torch.cat((keep_all, fresh), dim=-1)
        # History always receives the unregularized ranking, so the reservoir
        # cannot reinforce its own earlier intervention.  Sliding counts are
        # maintained by two scatter operations rather than replaying W*K ids.
        if layer_idx not in selection_frequency:
            shape = (
                cache.batch_size, cache.num_key_value_heads, cache.max_blocks
            )
            selection_frequency[layer_idx] = torch.zeros(
                shape, device=selected.device, dtype=torch.int32
            )
            selection_last_priority[layer_idx] = torch.zeros(
                shape, device=selected.device, dtype=torch.float32
            )
        frequency = selection_frequency[layer_idx]
        if len(history) == args.selection_history_window:
            expired = history.popleft()
            frequency.scatter_add_(
                -1, expired, -torch.ones_like(expired, dtype=frequency.dtype)
            )
        frequency.scatter_add_(
            -1, raw_selected,
            torch.ones_like(raw_selected, dtype=frequency.dtype),
        )
        rank_priority = (
            step * (raw_selected.shape[-1] + 1)
            + torch.arange(
                raw_selected.shape[-1], 0, -1,
                device=selected.device, dtype=torch.float32,
            )
        )
        selection_last_priority[layer_idx].scatter_(
            -1,
            raw_selected,
            rank_priority[None, None].expand_as(raw_selected),
        )
        history.append(raw_selected)
        if args.lightweight:
            return selected
        eligible = [block for block in target_blocks if first <= block < last]
        for head in range(cache.num_key_value_heads):
            before_order = [
                int(value) for value in selected[0, head].tolist()
            ]
            before = set(before_order)
            selected_rank = {
                block: rank for rank, block in enumerate(before_order, start=1)
            }
            evidence_counts = [
                int(cache.component_count[layer_idx, 0, head, block].item())
                for block in eligible
            ]
            # An unselected block has rank strictly below the retained set;
            # record K+1 as a lower bound without recomputing all router logits.
            evidence_ranks = [
                selected_rank.get(block, selected.shape[-1] + 1)
                for block in eligible
            ]
            if mode == "force_evidence":
                missing = [block for block in eligible if block not in before]
                cursor = selected.shape[-1] - 1
                protected = set(eligible)
                for block in missing:
                    while cursor >= 0 and int(selected[0, head, cursor]) in protected:
                        cursor -= 1
                    if cursor < 0:
                        break
                    selected[0, head, cursor] = block
                    cursor -= 1
            after = set(int(value) for value in selected[0, head].tolist())
            rows.append({
                "mode": mode,
                "step": step,
                "layer": int(layer_idx),
                "head": head,
                "eligible_evidence_blocks": len(eligible),
                "before_recall": (
                    sum(block in before for block in eligible) / len(eligible)
                    if eligible else 1.0
                ),
                "after_recall": (
                    sum(block in after for block in eligible) / len(eligible)
                    if eligible else 1.0
                ),
                "evidence_components_mean": (
                    sum(evidence_counts) / len(evidence_counts)
                    if evidence_counts else float("nan")
                ),
                "evidence_components_min": (
                    min(evidence_counts) if evidence_counts else -1
                ),
                "evidence_rank_best": (
                    min(evidence_ranks) if evidence_ranks else -1
                ),
                "evidence_rank_worst": (
                    max(evidence_ranks) if evidence_ranks else -1
                ),
            })
        if mode == "baseline":
            traces.append({
                "step": step,
                "layer": int(layer_idx),
                "query": query_states.detach().to(torch.bfloat16).cpu(),
                "selected_blocks": selected.detach().to(torch.int32).cpu(),
            })
        return selected

    cache._select_block_ids = wrapped_select
    results = []
    modes = ("baseline",) if args.baseline_only else (
        "baseline", "force_evidence"
    )
    for current_mode in modes:
        mode = current_mode
        call_count.clear()
        selection_history.clear()
        selection_frequency.clear()
        selection_last_priority.clear()
        prediction = llm.generate(
            input_ids, gen_len=gen_len, temperature=0.0, top_p=1.0,
            top_k=50, verbose=False,
        )[0]
        score = (
            multi_words(prediction, answers)
            if args.task == "cwe"
            else needle_score(prediction, answers[0])
        )
        results.append({"mode": mode, "prediction": prediction, "score": score})
        print(f"{mode}: score={score:.4f} prediction={prediction!r}", flush=True)
    cache._select_block_ids = original_select

    args.output.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    frame = pd.DataFrame(rows)
    if rows:
        frame.to_parquet(args.output / "evidence_recall.parquet", index=False)
        per_mode = frame.groupby("mode", as_index=False).agg(
            before_recall=("before_recall", "mean"),
            after_recall=("after_recall", "mean"),
        )
        per_mode.to_csv(args.output / "evidence_recall_summary.csv", index=False)
        torch.save(traces, args.output / "baseline_decode_trace.pt")
        recall_payload = per_mode.to_dict(orient="records")
    else:
        recall_payload = []
    payload = {
        "task": args.task,
        "sample_index": args.sample_index,
        "prompt_tokens": len(encoded.input_ids),
        "budget": args.budget,
        "mean_components": args.mean_components,
        "center_placement": args.center_placement,
        "query_mass_bank_size": args.query_mass_bank_size,
        "query_mass_window": args.query_mass_window,
        "query_mass_gqa_reduce": args.query_mass_gqa_reduce,
        "previous_selection_fraction": args.previous_selection_fraction,
        "selection_history_window": args.selection_history_window,
        "answers": answers,
        "evidence_blocks": target_blocks,
        "results": results,
        "evidence_recall": recall_payload,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
