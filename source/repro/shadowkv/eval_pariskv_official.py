#!/usr/bin/env python
"""Evaluate the authors' native ParisKV runtime on our benchmark rows.

This file is only a dataset/metric adapter.  Model execution, PolarANN
retrieval, 4-bit reranking, CPU offload, UVA gather, and sparse attention all
come directly from the external ParisKV checkout selected by ``--author-root``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from tqdm import tqdm


def git_output(root: str, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", root, *args], text=True
    ).strip()


def normalize_ground_truth(ground_truth):
    if isinstance(ground_truth, list) and len(ground_truth) == 1:
        return ground_truth[0]
    return ground_truth


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--author-root", default=os.environ.get("PARISKV_AUTHOR_ROOT")
    )
    parser.add_argument("--shadowkv-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--datalen", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--final-topk", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-stop", type=int, default=-1)
    parser.add_argument("--sink-size", type=int, default=32)
    parser.add_argument("--local-size", type=int, default=32)
    parser.add_argument("--update-interval", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not args.author_root or not os.path.isdir(args.author_root):
        parser.error("--author-root or PARISKV_AUTHOR_ROOT is required")
    if not os.path.isdir(args.shadowkv_root):
        parser.error("--shadowkv-root must point to the ShadowKV checkout")
    if not os.path.isdir(args.model):
        parser.error("--model must be a local Qwen checkpoint")

    # The author checkout currently exposes its native sparse runtime through
    # QwenModel.  Refuse unsupported architectures instead of silently running
    # a different implementation.
    from transformers import AutoConfig

    model_type = AutoConfig.from_pretrained(
        args.model, trust_remote_code=True
    ).model_type
    if model_type not in {"qwen2", "qwen3"}:
        raise SystemExit(
            f"ParisKV official QwenModel does not support model_type={model_type!r}"
        )

    sys.path.insert(0, args.author_root)
    sys.path.insert(0, args.shadowkv_root)
    from model_hub.qwen import QwenModel
    from data.dataset import Dataset

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    author_commit = git_output(args.author_root, "rev-parse", "HEAD")
    author_dirty = bool(git_output(args.author_root, "status", "--porcelain"))

    llm = QwenModel(
        model_name=args.model,
        max_length=args.datalen + 2048,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    dataset = Dataset(
        f"ruler/{args.task}", llm.tokenizer, args.datalen, args.num_samples
    )
    stop = None if args.sample_stop < 0 else args.sample_stop
    prompts = dataset.tokenized_prompts[args.sample_start:stop]
    ground_truths = dataset.gt[args.sample_start:stop]
    if not prompts:
        raise SystemExit("selected sample shard is empty")

    codebook = os.path.join(
        args.author_root,
        "turboquant",
        "codebooks",
        "codebook_d128_m8_Kr1_Kw256_rabitq_sign.json",
    )
    polar_config = {
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")
    scores: list[float] = []
    for local_index, (prompt, ground_truth) in enumerate(
        tqdm(zip(prompts, ground_truths), total=len(prompts), desc="ParisKV official")
    ):
        prompt = prompt.to(device)
        mask = torch.ones_like(prompt)
        generated_ids, timing = llm.generate(
            attention_type="PolarANN",
            inputs_ids=prompt,
            attention_masks=mask,
            max_new_length=dataset.gen_len,
            attn_config=polar_config,
            temperature=0.0,
        )
        prediction = llm.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        ground_truth = normalize_ground_truth(ground_truth)
        score = float(dataset.metric(prediction, ground_truth))
        scores.append(score)
        record = {
            "sample_index": args.sample_start + local_index,
            "prediction": [prediction],
            "ground_truth": [ground_truth],
            "correct": scores.copy(),
            "avg_score": sum(scores) / len(scores),
            "timing": timing,
            "runtime": "ParisKV-author",
            "author_commit": author_commit,
            "author_dirty": author_dirty,
        }
        with out_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"PARISKV OFFICIAL QUALITY task={args.task} samples={len(scores)} "
        f"accuracy={sum(scores) / len(scores):.4f} author_commit={author_commit}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
