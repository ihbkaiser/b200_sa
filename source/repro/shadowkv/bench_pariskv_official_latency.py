#!/usr/bin/env python
"""Isolated 128K latency measurement for the authors' ParisKV runtime.

The loop mirrors ``bench_latency.py``: load the model once, run complete
untimed warm-up passes, then report median prefill and decode latency over
measured passes.  It deliberately imports ParisKV from the external authors'
checkout instead of the ShadowKV integration tree.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-root", default=os.environ.get("PARISKV_AUTHOR_ROOT"))
    parser.add_argument("--model", default=os.environ.get("SHADOWKV_QWEN3_PATH"))
    parser.add_argument("--input-tokens", type=int, default=131072)
    parser.add_argument("--decode-steps", type=int, default=33)
    parser.add_argument("--final-topk", type=int, default=4096)
    parser.add_argument("--sink-size", type=int, default=32)
    parser.add_argument("--local-size", type=int, default=32)
    parser.add_argument("--update-interval", type=int, default=512)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ruler-jsonl", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    if not args.author_root or not os.path.isdir(args.author_root):
        parser.error("--author-root or PARISKV_AUTHOR_ROOT is required")
    if not args.model or not os.path.isdir(args.model):
        parser.error("--model or SHADOWKV_QWEN3_PATH must be a local model")
    if not os.path.isfile(args.ruler_jsonl):
        parser.error(f"RULER row does not exist: {args.ruler_jsonl}")

    sys.path.insert(0, args.author_root)
    from model_hub.qwen import QwenModel

    codebook = os.path.join(
        args.author_root,
        "turboquant/codebooks/codebook_d128_m8_Kr1_Kw256_rabitq_sign.json",
    )
    torch.cuda.set_device(torch.device(args.device))
    model = QwenModel(
        model_name=args.model,
        max_length=args.input_tokens + args.decode_steps + 8,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    with open(args.ruler_jsonl, encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    ids = model.tokenizer.encode(row["input"], add_special_tokens=False)
    ids = ids[: args.input_tokens]
    if len(ids) < args.input_tokens:
        ids = (ids * math.ceil(args.input_tokens / len(ids)))[: args.input_tokens]
    input_ids = torch.tensor(ids, device=args.device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    config = {
        "PolarANN": {
            "sink_size": args.sink_size,
            "local_size": args.local_size,
            "core": 22,
            "nprobe": 150,
            "cache_unit_size": 8,
            "cache_cluster_num": 450,
            "dynamic_update_interval": args.update_interval,
            "final_topk": args.final_topk,
            "enable_offload": True,
            "codebook_path": codebook,
        }
    }

    def one_pass():
        _, stats = model.generate(
            attention_type="PolarANN",
            inputs_ids=input_ids,
            attention_masks=attention_mask.clone(),
            max_new_length=args.decode_steps,
            attn_config=config,
            temperature=0.0,
            disable_early_stop=True,
        )
        torch.cuda.synchronize()
        return stats

    for _ in range(args.warmup_passes):
        one_pass()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    measured = [one_pass() for _ in range(args.reps)]
    record = {
        "runtime": "ParisKV-author",
        "author_commit": subprocess.check_output(
            ["git", "-C", args.author_root, "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": "qwen3",
        "datalen": args.input_tokens,
        "method": "pariskv_official",
        "sparse_budget": args.final_topk,
        "sink_tokens": args.sink_size,
        "recent_tokens": args.local_size,
        "prefill_s": statistics.median(x["prefill_time"] for x in measured),
        "decode_ms": statistics.median(x["avg_latency_ms"] for x in measured),
        "decode_tokens_per_s": statistics.median(x["throughput"] for x in measured),
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "warmup_passes": args.warmup_passes,
        "reps": args.reps,
        "decode_steps": args.decode_steps,
        "ruler_jsonl": os.path.abspath(args.ruler_jsonl),
    }
    print(json.dumps(record, indent=2))
    if args.out:
        with open(args.out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
