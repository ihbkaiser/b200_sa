#!/usr/bin/env python
"""
Fit M51's PQ codebook ONCE, offline, and stop paying for it per prompt.

Profiling put 90% of M51's per-context metadata cost in the k-means that
refits this codebook on every prompt (0.918s of 1.016s per layer at 32K, ~33s
across the model). The tangents it quantises are softmax-weighted key means, so
they live in key space, which is far more stable per (layer, kv head) across
contexts than per prompt -- which makes an offline codebook worth trying before
anyone writes a kernel for the k-means.

Calibration comes from LongBench documents, so nothing here has seen a RULER
evaluation context.

  source repro/shadowkv/env_m1.sh
  CUDA_VISIBLE_DEVICES=0 $PY repro/shadowkv/m51_fit_codebook.py \
      --anchors $SHADOWKV_M51_ANCHORS_QWEN3 --docs 4 --context 16384
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ShadowKV"))

LONGBENCH = ("/storage/baonn/huggingface/hub/datasets--Xnhyacinth--LongBench/"
             "snapshots/2e9ade51ebf45d98942056c0716234f9d5d257d5")


def longbench_contexts(tasks, n, tokenizer, context_tokens):
    import pyarrow.parquet as pq
    out = []
    for task in tasks:
        files = sorted(glob.glob(os.path.join(LONGBENCH, task, "*.parquet")))
        if not files:
            continue
        table = pq.read_table(files[0])
        ctxs = [c.as_py() for c in table.column("context")]
        ctxs.sort(key=len, reverse=True)          # longest first: we need full windows
        for c in ctxs:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) >= context_tokens:
                out.append(torch.tensor(ids[:context_tokens]).unsqueeze(0))
                break
        if len(out) >= n:
            break
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default=os.environ.get("SHADOWKV_M51_ANCHORS_QWEN3"))
    ap.add_argument("--model", default=os.environ.get("SHADOWKV_QWEN3_PATH"))
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--codes", type=int, default=64)
    ap.add_argument("--subdim", type=int, default=8)
    ap.add_argument("--out", default=None, help="default: alongside --anchors, with the codebook added")
    args = ap.parse_args()

    from models import choose_model_class
    from models.routed_pq_cache import _batched_kmeans

    LLM = choose_model_class(args.model)
    llm = LLM(model_name=args.model, batch_size=1, device="cuda:0",
              max_length=args.context + 2048, attn_mode="m51", dtype=torch.bfloat16,
              m51_anchors=args.anchors, m51_pq_mode="per_context", dense_layers=0)
    cache = llm.kv_cache
    cache.tangent_sink = []

    tasks = ["gov_report", "narrativeqa", "multi_news", "qmsum", "2wikimqa", "hotpotqa"]
    ctxs = longbench_contexts(tasks, args.docs, llm.tokenizer, args.context)
    if len(ctxs) < args.docs:
        raise SystemExit(f"only found {len(ctxs)} LongBench documents of >= {args.context} tokens")
    print(f"calibrating on {len(ctxs)} LongBench documents of {args.context} tokens each")

    for i, ids in enumerate(ctxs):
        t0 = time.perf_counter()
        llm.prefill(ids.to("cuda:0"))
        print(f"  doc {i}: prefill {time.perf_counter()-t0:.1f}s, "
              f"{len(cache.tangent_sink)} layer-samples collected")

    n_layers = cache.num_layers
    H, D = cache.num_key_value_heads, cache.head_dim
    n_sub = D // args.subdim

    pooled = {}
    for layer, sample in cache.tangent_sink:
        pooled.setdefault(layer, []).append(sample)
    del cache.tangent_sink

    codebook = np.zeros((n_layers, H, n_sub, args.codes, args.subdim), dtype=np.float32)
    err = []
    for layer in range(n_layers):
        x = torch.cat(pooled[layer], dim=1).to("cuda:0", torch.float32)     # [H, N, D]
        for sub in range(n_sub):
            sl = slice(sub * args.subdim, (sub + 1) * args.subdim)
            cb, code = _batched_kmeans(x[:, :, sl], args.codes)
            codebook[layer, :, sub] = cb.cpu().numpy()
            recon = torch.gather(cb, 1, code.unsqueeze(-1).expand(-1, -1, args.subdim))
            err.append(float(torch.linalg.vector_norm(x[:, :, sl] - recon, dim=-1).mean()))
        del x
        torch.cuda.empty_cache()
        if layer % 6 == 0:
            print(f"  layer {layer}/{n_layers}")

    out = args.out or args.anchors
    z = dict(np.load(args.anchors))
    z["pq_codebook"] = codebook
    z["pq_calibration"] = np.array(
        f"longbench:{','.join(tasks[:args.docs])} docs={args.docs} ctx={args.context}")
    np.savez(out, **z)
    print(f"\nwrote {out}")
    print(f"  codebook {codebook.shape}, mean in-sample subspace residual {np.mean(err):.4f}")
    print("  in-sample residual is a floor, not a guarantee -- what matters is the")
    print("  RULER accuracy and coverage this codebook gives on unseen contexts.")


if __name__ == "__main__":
    main()
