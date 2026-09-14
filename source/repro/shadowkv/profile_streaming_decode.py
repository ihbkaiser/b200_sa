#!/usr/bin/env python
"""Component-level decode profile for the canonical streaming router.

The profiler uses nested CUDA events around the existing production path.  It
does not replace any kernel or change selection.  Reported ``selection`` time
is retrieval minus the nested scorer, while ``other`` is the complete decode
step minus update/retrieval/gather/attention and therefore includes the model
projections, MLP, LM head and any launch gaps.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ShadowKV"))

import torch  # noqa: E402

from bench_latency import MODELS, real_ruler_prompt, require_idle_gpu  # noqa: E402


class EventProfile:
    def __init__(self) -> None:
        self.active = False
        self.current_step = -1
        self.events: dict[
            str, list[tuple[int, torch.cuda.Event, torch.cuda.Event]]
        ] = (
            defaultdict(list)
        )

    def call(self, name, fn, *args, **kwargs):
        if not self.active:
            return fn(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        self.events[name].append((self.current_step, start, end))
        return result

    def wrap_instance(self, instance, attribute: str, name: str) -> None:
        original = getattr(instance, attribute)

        def wrapped(*args, **kwargs):
            return self.call(name, original, *args, **kwargs)

        setattr(instance, attribute, wrapped)

    def milliseconds(self, name: str) -> float:
        return sum(start.elapsed_time(end) for _, start, end in self.events[name])

    def milliseconds_by_step(self, name: str, count: int) -> list[float]:
        result = [0.0] * count
        for step, start, end in self.events[name]:
            result[step] += start.elapsed_time(end)
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="qwen3", choices=sorted(MODELS))
    parser.add_argument("--datalen", type=int, required=True)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--extra-fraction", type=float, default=0.25)
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--recent", type=int, default=32)
    parser.add_argument(
        "--update-interval", type=int, default=0,
        help="batch-flush cadence in tokens; 0 keeps the per-block lifecycle",
    )
    parser.add_argument("--task", default="niah_multikey_2")
    parser.add_argument(
        "--variant",
        choices=("no_rerank", "rank16_int4_2b", "rank16_int4_4b", "rank16_int4_direct"),
        default="no_rerank",
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=16)
    parser.add_argument("--router-backend", choices=["torch", "triton"], default="triton")
    parser.add_argument("--gather-backend", choices=["uva", "torch"], default="uva")
    parser.add_argument(
        "--backing-store",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Location of the exact KV cache; metadata remains on GPU.",
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.block_size != 8 or args.extra_fraction != 0.25:
        raise SystemExit("efficiency gate is locked to block=8, extra_fraction=0.25")

    require_idle_gpu()
    refine = args.variant != "no_rerank"
    refine_factor = {
        "no_rerank": 1.0,
        "rank16_int4_2b": 2.0,
        "rank16_int4_4b": 4.0,
        "rank16_int4_direct": 1.0,
    }[args.variant]
    refine_ratio = 1.0 if args.variant == "rank16_int4_direct" else None
    os.environ["STREAMING_REFINE_SKETCH_RANK"] = "16" if refine else "0"
    os.environ["STREAMING_REFINE_SKETCH_BITS"] = "4"
    os.environ["STREAMING_REFINE_SKETCH_BASIS"] = "query_aware"
    os.environ.setdefault("SHADOWKV_CENTER_PLACEMENT", "robust_trimmed")
    os.environ.setdefault("SHADOWKV_CENTER_ALLOCATION", "tail_cvar")
    os.environ.setdefault("SHADOWKV_ROBUST_QUERY_SOURCE", "self_k")
    os.environ.setdefault("SHADOWKV_ROBUST_TRIM_FRACTION", "0.25")
    os.environ.setdefault("SHADOWKV_TAIL_CVAR_FRACTION", "0.25")
    os.environ.setdefault("SHADOWKV_TAIL_GAP_CORRECTION_SCALE", "0.25")
    os.environ.setdefault("SHADOWKV_CENTER_DISPERSION_CORRECTION", "0")
    from models import choose_model_class
    import models.base as base_module

    model_path = os.environ[MODELS[args.model_key]]
    llm_class = choose_model_class(model_path)
    llm = llm_class(
        model_name=model_path,
        batch_size=1,
        device="cuda:0",
        max_length=args.datalen + args.warmup_steps + args.measure_steps + 16,
        attn_mode="adaptive_centroid_lse_streaming_prefix4_querymean",
        dtype=torch.bfloat16,
        sparse_budget=args.budget,
        rank=160,
        chunk_size=args.block_size,
        dense_layers=0,
        group_reduce="max",
        quest_prefix_tokens=args.prefix,
        quest_recent_tokens=args.recent,
        streaming_update_interval=args.update_interval or None,
        router_split_fraction=args.extra_fraction,
        self_lse_temperatures=(1.0,),
        streaming_offload=args.backing_store == "cpu",
        streaming_gather_backend=args.gather_backend,
        streaming_router_backend=args.router_backend,
        streaming_refine_factor=refine_factor,
        streaming_refine_candidate_ratio=refine_ratio,
        streaming_refine_tokens=refine,
        streaming_max_components=8,
        streaming_compact_metadata=True,
        streaming_center_bits=8,
    )
    ids = real_ruler_prompt(llm, args.datalen, args.model_key, args.task)
    torch.cuda.synchronize()
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    prefill_start.record()
    logits = llm.prefill(ids)
    prefill_end.record()
    llm.kv_cache.H2D()
    torch.cuda.synchronize()
    prefill_ms = prefill_start.elapsed_time(prefill_end)
    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

    profile = EventProfile()
    cache = llm.kv_cache
    profile.wrap_instance(cache, "update_kv_cache", "update")
    profile.wrap_instance(cache, "_refine_sketch_logits", "rerank")
    if refine:
        # Token refinement bypasses _select_block_ids: it obtains the coarse
        # logits directly and performs candidate reranking inside the ragged
        # position-id path.  Time the actual public retrieval call while
        # retaining nested score/rerank events for a clean subtraction.
        profile.wrap_instance(cache, "_block_logits", "score")
        profile.wrap_instance(cache, "get_retrieval_position_ids", "retrieval")
    else:
        # The canonical query-mean selector ranks raw logits directly because
        # a singleton-query softmax cannot change top-k.  Instrument the
        # common scorer below that optimization rather than the now-bypassed
        # probability wrapper.
        profile.wrap_instance(cache, "_block_logits", "score")
        profile.wrap_instance(cache, "_select_block_ids", "retrieval")
    profile.wrap_instance(cache, "get_key_value_cache", "gather")
    original_attention = base_module.flash_attn_with_kvcache

    def profiled_attention(*call_args, **call_kwargs):
        return profile.call("attention", original_attention, *call_args, **call_kwargs)

    base_module.flash_attn_with_kvcache = profiled_attention

    def decode_once():
        nonlocal next_token
        output = llm.inference(
            input_ids=next_token, position_ids=llm.get_ctx(next_token)
        )
        next_token = output[:, -1].argmax(dim=-1, keepdim=True)

    for _ in range(args.warmup_steps):
        decode_once()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    step_events = []
    sealed_steps = []
    profile.active = True
    for step in range(args.measure_steps):
        profile.current_step = step
        before_blocks = cache.block_state[0].sealed_blocks
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        decode_once()
        end.record()
        step_events.append((start, end))
        sealed_steps.append(cache.block_state[0].sealed_blocks > before_blocks)
    profile.active = False
    torch.cuda.synchronize()

    step_ms = [start.elapsed_time(end) for start, end in step_events]
    total = sum(step_ms)
    raw = {
        name: profile.milliseconds_by_step(name, args.measure_steps)
        for name in (
            "update", "score", "rerank", "retrieval", "gather", "attention"
        )
    }
    per_step = []
    for step, elapsed in enumerate(step_ms):
        selection = max(
            0.0,
            raw["retrieval"][step]
            - raw["score"][step]
            - raw["rerank"][step],
        )
        other = max(
            0.0,
            elapsed
            - raw["update"][step]
            - raw["retrieval"][step]
            - raw["gather"][step]
            - raw["attention"][step],
        )
        per_step.append(
            {
                "step": step,
                "sealed_block": sealed_steps[step],
                "total": elapsed,
                "update": raw["update"][step],
                "score": raw["score"][step],
                "rerank": raw["rerank"][step],
                "selection_after_score": selection,
                "gather": raw["gather"][step],
                "attention": raw["attention"][step],
                "other_model_and_gaps": other,
            }
        )
    component_names = (
        "update",
        "score",
        "rerank",
        "selection_after_score",
        "gather",
        "attention",
        "other_model_and_gaps",
    )
    component_totals = {
        name: sum(row[name] for row in per_step) for name in component_names
    }
    nonsealed = [row for row in per_step if not row["sealed_block"]]
    sealed = [row for row in per_step if row["sealed_block"]]
    result = {
        "model": args.model_key,
        "datalen": args.datalen,
        "budget": args.budget,
        "variant": args.variant,
        "block_size": args.block_size,
        "active_mean_centers": 1.0 + args.extra_fraction,
        "prefix": args.prefix,
        "recent": args.recent,
        "router_backend": args.router_backend,
        "gather_backend": args.gather_backend,
        "backing_store": args.backing_store,
        "update_interval": args.update_interval,
        "temporal_block_reuse": cache._can_reuse_selected_blocks(),
        "prefill_ms": prefill_ms,
        "decode_median_ms": statistics.median(step_ms),
        "decode_min_ms": min(step_ms),
        "decode_max_ms": max(step_ms),
        "measured_steps": args.measure_steps,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "component_ms_per_token": {
            key: value / args.measure_steps
            for key, value in component_totals.items()
        },
        "component_fraction": {
            key: value / total for key, value in component_totals.items()
        },
        "component_median_ms": {
            name: statistics.median(row[name] for row in per_step)
            for name in component_names
        },
        "nonsealed_median_ms": {
            name: statistics.median(row[name] for row in nonsealed)
            for name in ("total", *component_names)
        },
        "sealed_median_ms": (
            {
                name: statistics.median(row[name] for row in sealed)
                for name in ("total", *component_names)
            }
            if sealed
            else None
        ),
        "per_step": per_step,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
