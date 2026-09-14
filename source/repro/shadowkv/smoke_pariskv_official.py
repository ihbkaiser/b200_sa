#!/usr/bin/env python
"""Run the authors' native ParisKV inference path on a local Qwen model.

This exercises the authors' collision, radix-topk, 4-bit rerank, pinned-CPU
cache, UVA gather, and attention kernels.  Quality evaluation uses the same
native runtime through ``eval_pariskv_official.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--author-root", default=os.environ.get("PARISKV_AUTHOR_ROOT")
    )
    parser.add_argument(
        "--model", default=os.environ.get("SHADOWKV_QWEN3_PATH")
    )
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--new-tokens", type=int, default=2)
    parser.add_argument("--final-topk", type=int, default=512)
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument("--local-size", type=int, default=512)
    parser.add_argument("--update-interval", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out")
    args = parser.parse_args()
    if not args.author_root or not os.path.isdir(args.author_root):
        parser.error("--author-root or PARISKV_AUTHOR_ROOT is required")
    if not args.model or not os.path.isdir(args.model):
        parser.error("--model or SHADOWKV_QWEN3_PATH must be a local model")
    if args.input_tokens <= args.final_topk + args.sink_size + args.local_size:
        parser.error("input must exceed final_topk + sink + local regions")

    sys.path.insert(0, args.author_root)
    from model_hub.qwen import QwenModel

    codebook = os.path.join(
        args.author_root,
        "turboquant",
        "codebooks",
        "codebook_d128_m8_Kr1_Kw256_rabitq_sign.json",
    )
    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = QwenModel(
        model_name=args.model,
        max_length=args.input_tokens + args.new_tokens + 8,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    seed = model.tokenizer.encode(
        "ParisKV native offload and retrieval kernel smoke test. ",
        add_special_tokens=False,
    )
    repeats = math.ceil(args.input_tokens / len(seed))
    ids = torch.tensor(
        (seed * repeats)[: args.input_tokens],
        device=args.device,
        dtype=torch.long,
    ).unsqueeze(0)
    mask = torch.ones_like(ids)
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
    generated, stats = model.generate(
        attention_type="PolarANN",
        inputs_ids=ids,
        attention_masks=mask,
        max_new_length=args.new_tokens,
        attn_config=config,
        temperature=0.0,
        disable_early_stop=True,
    )
    torch.cuda.synchronize()
    result = {
        "runtime": "ParisKV-author",
        "author_commit": subprocess.check_output(
            ["git", "-C", args.author_root, "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": args.model,
        "input_tokens": args.input_tokens,
        "final_topk": args.final_topk,
        "sink_size": args.sink_size,
        "local_size": args.local_size,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "generated_tokens": len(generated[0]),
        **stats,
    }
    print("PARISKV OFFICIAL E2E PASS")
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
