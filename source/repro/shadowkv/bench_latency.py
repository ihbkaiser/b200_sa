#!/usr/bin/env python
"""
Prefill latency and single-decode-step latency, one method per process.

  source repro/shadowkv/env_m1.sh
  CUDA_VISIBLE_DEVICES=6 $PY repro/shadowkv/bench_latency.py \
      --model_key llama32 --datalen 32768 --method quest_streaming --sparse_budget 2048

This is NOT the campaign's per-cell wall time. That number is dominated by
model loading and by 96 prefills with only ~100 generated tokens, which is
close to a pure prefill benchmark -- exactly the phase where all three methods
run the same dense flash-attention. The methods differ in DECODE, so decode
gets measured on its own here.

Measurement discipline, because a speed number taken carelessly is worth less
than no number:
  * one method per process -- peak memory across methods in one process is an
    artifact of the allocator, not of the method;
  * refuses to run if any other process holds the GPU (sharing a card moves
    latency by 3-5x, enough to invert a fast/slow conclusion);
  * warmup before timing, cuda synchronize around every measured region,
    the first decode step discarded (it pays for lazy allocation),
    median over repeats rather than mean.
"""

import argparse
import importlib.metadata
import json
import os
import resource
import statistics
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ShadowKV"))

import torch  # noqa: E402

MODELS = {"llama32": "SHADOWKV_LLAMA32_PATH", "qwen3": "SHADOWKV_QWEN3_PATH"}
TEMPLATE = {"llama32": "llama-3", "qwen3": "qwen"}


def percentile(values, q):
    """Linearly interpolated percentile without a numpy dependency."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def real_ruler_prompt(llm, datalen, model_key=None, task="niah_multikey_2"):
    """One real RULER prompt, trimmed/padded to exactly datalen tokens."""
    import json
    name = TEMPLATE.get(model_key, "qwen")
    data_root = os.environ.get(
        "SHADOWKV_RULER_DATA_ROOT",
        os.path.join(REPO, "ShadowKV", "data", "ruler", "data"),
    )
    path = os.path.join(data_root, name, str(datalen), task, "validation.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"no RULER data at {path}; build it with build_ruler.sh")
    with open(path) as f:
        row = json.loads(f.readline())
    ids = llm.tokenizer.encode(row["input"], add_special_tokens=False)[:datalen]
    if len(ids) < datalen:
        ids = ids + ids[:datalen - len(ids)]
    return torch.tensor(ids, device="cuda:0").unsqueeze(0)


def require_idle_gpu():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is None:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to one physical GPU id")
    out = subprocess.check_output(
        ["nvidia-smi", "--id=" + vis, "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip()
    if out:
        raise SystemExit(f"GPU {vis} is busy (pids: {out.splitlines()}) -- "
                         f"a shared card moves latency by 3-5x; pick an idle one")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", required=True, choices=sorted(MODELS))
    ap.add_argument("--datalen", type=int, required=True)
    ap.add_argument("--method", required=True,
                    choices=[
                        "full", "quest_streaming", "shadowkv", "shadowkv_cpu", "m51",
                        "pariskv_author_common",
                        "pqcache_author_common", "pqcache_author_native",
                        "magicpig_author_common",
                        "infllm_author_common",
                        "adaptive_centroid_lse_streaming_prefix4",
                        "adaptive_centroid_lse_streaming_prefix4_querymean",
                    ])
    ap.add_argument("--sparse_budget", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=160)
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--shadow_outlier_chunks", type=int, default=48)
    ap.add_argument("--page_size", type=int, default=16)
    ap.add_argument("--dense_layers", type=int, default=2)
    ap.add_argument("--prefix_tokens", type=int, default=32)
    ap.add_argument("--recent_tokens", type=int, default=32)
    ap.add_argument("--prefill_reps", type=int, default=3)
    ap.add_argument("--decode_steps", type=int, default=33, help="first step is discarded")
    ap.add_argument("--warmup_passes", type=int, default=1,
                    help="complete untimed prefill/decode passes")
    ap.add_argument("--m51_anchors", default=None)
    ap.add_argument("--m51_select_mode", default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--m51_pq_subdim", type=int, default=8)
    ap.add_argument("--m51_rank_anchors", type=int, default=1,
                    help="0 = min over every anchor, 1 = the routed branch only")
    ap.add_argument("--m51_rank_reduce", default="max", choices=["max", "softmax"])
    ap.add_argument("--m51_coverage", type=float, default=0.90)
    ap.add_argument("--m51_leaf", type=int, default=8)
    ap.add_argument("--m51_anchors_n", type=int, default=8)
    ap.add_argument("--m51_pq_mode", choices=["offline", "per_context", "warm"], default="offline")
    ap.add_argument("--m51_pq_warm_iters", type=int, default=2)
    ap.add_argument("--m51_group_select", choices=["shared", "per_query"], default="per_query")
    ap.add_argument("--m51_decode_mode", choices=["gather", "dense_mask"], default="dense_mask")
    ap.add_argument("--m51_load_batch", type=int, default=128)
    ap.add_argument("--real_text", action="store_true", default=False,
                    help="prefill a real RULER prompt; required for data-adaptive methods")
    ap.add_argument("--ruler_task", default="niah_multikey_2")
    ap.add_argument("--streaming_offload", action="store_true",
                    help="keep exact streaming K/V in pinned CPU memory")
    ap.add_argument("--streaming_gather_backend", choices=["auto", "uva", "torch"],
                    default="auto")
    ap.add_argument("--streaming_router_backend", choices=["torch", "triton"],
                    default="torch")
    ap.add_argument("--pariskv_author_root",
                    default=os.environ.get("PARISKV_AUTHOR_ROOT"))
    ap.add_argument("--extra_fraction", type=float, default=0.25)
    ap.add_argument("--out", default=None, help="append one json line here")
    args = ap.parse_args()

    require_idle_gpu()
    model_path = os.environ[MODELS[args.model_key]]
    querymean_default = args.method.endswith("_querymean")
    resolved_center_allocation = os.environ.get(
        "SHADOWKV_CENTER_ALLOCATION",
        "tail_absolute_rate_distortion" if querymean_default else "self_lse",
    )

    from models import choose_model_class
    LLM = choose_model_class(model_path)

    model_dtype = (
        torch.float16 if args.method == "pqcache_author_native"
        else torch.bfloat16
    )
    kw = dict(model_name=model_path, batch_size=1, device="cuda:0",
              max_length=args.datalen + 2048, attn_mode=args.method,
              dtype=model_dtype, sparse_budget=args.sparse_budget,
              rank=args.rank, chunk_size=args.chunk_size)
    if args.method == "quest_streaming":
        kw.update(
            page_size=args.page_size, dense_layers=args.dense_layers,
            group_reduce="max",
            quest_prefix_tokens=args.prefix_tokens,
            quest_recent_tokens=args.recent_tokens,
            streaming_offload=args.streaming_offload,
            streaming_gather_backend=args.streaming_gather_backend,
        )
    if args.method in {"shadowkv", "shadowkv_cpu"}:
        kw.update(shadow_outlier_chunks=args.shadow_outlier_chunks)
    if args.method == "pariskv_author_common":
        kw.update(
            dense_layers=0, group_reduce="max",
            quest_prefix_tokens=args.prefix_tokens,
            quest_recent_tokens=args.recent_tokens,
            streaming_offload=args.streaming_offload,
            streaming_gather_backend=args.streaming_gather_backend,
            pariskv_author_root=args.pariskv_author_root,
        )
    if args.method in {
        "pqcache_author_common", "pqcache_author_native",
        "magicpig_author_common",
        "infllm_author_common",
    }:
        # Author roots and method-specific native knobs are deliberately read
        # from the environment by the adapters so the same commit-stamped
        # configuration is shared with run_cell.sh.
        pass
    if args.method == "m51":
        kw.update(m51_anchors=args.m51_anchors, m51_coverage=args.m51_coverage,
                  m51_leaf=args.m51_leaf, m51_anchors_n=args.m51_anchors_n,
                  m51_pq_mode=args.m51_pq_mode, m51_pq_warm_iters=args.m51_pq_warm_iters, m51_group_select=args.m51_group_select,
                  m51_decode_mode=args.m51_decode_mode, m51_load_batch=args.m51_load_batch,
                  m51_select_mode=args.m51_select_mode, m51_budget=args.sparse_budget,
                  m51_pq_subdim=args.m51_pq_subdim, dense_layers=0,
                  m51_rank_anchors=args.m51_rank_anchors,
                  m51_rank_reduce=args.m51_rank_reduce)
    if args.method.startswith("adaptive_centroid_lse_streaming_prefix4"):
        kw.update(
            dense_layers=0, group_reduce="max",
            quest_prefix_tokens=args.prefix_tokens,
            quest_recent_tokens=args.recent_tokens,
            router_split_fraction=args.extra_fraction,
            self_lse_temperatures=(1.0,),
            streaming_offload=args.streaming_offload,
            streaming_gather_backend=args.streaming_gather_backend,
            streaming_router_backend=args.streaming_router_backend,
            streaming_max_components=8,
            streaming_compact_metadata=True,
            streaming_center_bits=8,
        )
    llm = LLM(**kw)

    # Random ids are fine for a dense kernel, whose work does not depend on the
    # token content. They are NOT fine for a method whose LOAD DEPTH is decided
    # by the data: unstructured context spreads attention mass evenly, so M51's
    # stopping rule reads 87% of blocks on noise against ~20% on RULER -- which
    # corrupts its latency, not just its traffic. --real_text prefills an actual
    # RULER prompt instead, and is the default for any budget-adaptive method.
    if args.real_text:
        ids = real_ruler_prompt(llm, args.datalen, args.model_key, args.ruler_task)
    else:
        torch.manual_seed(0)
        ids = torch.randint(0, llm.vocab_size if hasattr(llm, "vocab_size") else 30000,
                            (1, args.datalen), device="cuda:0")

    def one_pass(measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = llm.prefill(ids)
        # A prefill result is operationally usable only when async KV offload,
        # index/summary construction and required host/device transfers have
        # completed. Stopping before this barrier makes an asynchronous method
        # look artificially cheap and shifts its cost into the first decode
        # step, which this benchmark deliberately discards.
        llm.kv_cache.H2D()
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0

        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        steps = []
        for i in range(args.decode_steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = llm.inference(input_ids=nxt, position_ids=llm.get_ctx(nxt))
            torch.cuda.synchronize()
            steps.append(time.perf_counter() - t0)
            nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
        return prefill_s, steps[1:] if len(steps) > 1 else steps

    for _ in range(args.warmup_passes):
        one_pass(False)
    torch.cuda.reset_peak_memory_stats()

    prefills, decodes = [], []
    for _ in range(args.prefill_reps):
        p, d = one_pass(True)
        prefills.append(p)
        decodes.extend(d)

    peak = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    extra = {}
    if hasattr(llm.kv_cache, "traffic"):
        extra = llm.kv_cache.traffic() or {}
    actual_mean_centers = None
    if hasattr(llm.kv_cache, "component_count"):
        component_sum = 0
        block_count = 0
        for layer_idx, state in enumerate(llm.kv_cache.block_state):
            first, last = state.candidate_block_range
            counts = llm.kv_cache.component_count[
                layer_idx, :, :, first:last
            ]
            component_sum += int(counts.sum().item())
            block_count += counts.numel()
        if block_count:
            actual_mean_centers = component_sum / block_count
    variant = ""
    if args.method == "m51":
        variant = f"{args.m51_pq_mode}/{args.m51_group_select}/{args.m51_decode_mode}"
    rec = dict(model=args.model_key, datalen=args.datalen,
               ruler_task=(args.ruler_task if args.real_text else None),
               method=args.method, variant=variant,
               model_dtype=str(model_dtype),
               torch_version=torch.__version__,
               cuda_version=torch.version.cuda,
               flash_attn_version=importlib.metadata.version("flash-attn"),
               gpu_name=torch.cuda.get_device_name("cuda:0"),
               rmsnorm_backend=os.environ.get("SHADOWKV_RMSNORM_BACKEND", "torch"),
               center_allocation=resolved_center_allocation,
               configured_mean_centers=(
                   None if resolved_center_allocation
                   == "tail_absolute_rate_distortion"
                   else 1.0 + args.extra_fraction
               ),
               absolute_rd_penalty=(
                   float(os.environ.get("SHADOWKV_ABSOLUTE_RD_PENALTY", "1.5"))
                   if resolved_center_allocation
                   == "tail_absolute_rate_distortion" else None
               ),
               actual_mean_centers=actual_mean_centers,
               temporal_block_reuse=(
                   args.streaming_offload
                   and args.streaming_gather_backend in {"auto", "uva"}
                   and (
                       args.method == "quest_streaming"
                       or args.method.startswith(
                           "adaptive_centroid_lse_streaming_prefix4"
                       )
                   )
               ),
               real_text=bool(args.real_text),
               m51_pq_mode=args.m51_pq_mode, m51_anchors_n=args.m51_anchors_n,
               m51_select_mode=args.m51_select_mode,
               m51_rank_anchors=args.m51_rank_anchors,
               m51_rank_reduce=args.m51_rank_reduce,
               coverage=(args.m51_coverage if args.method == "m51" else None),
               sparse_budget=(0 if args.method == "full" else args.sparse_budget),
               shadow_outlier_chunks=(
                   args.shadow_outlier_chunks
                   if args.method in {"shadowkv", "shadowkv_cpu"} else None
               ),
               prefill_s=statistics.median(prefills),
               prefill_mean_s=statistics.mean(prefills),
               prefill_min_s=min(prefills),
               prefill_max_s=max(prefills),
               decode_ms=statistics.median(decodes) * 1000,
               decode_mean_ms=statistics.mean(decodes) * 1000,
               decode_p95_ms=percentile(decodes, 0.95) * 1000,
               decode_ms_min=min(decodes) * 1000,
               decode_ms_max=max(decodes) * 1000,
               peak_gib=peak, peak_reserved_gib=peak_reserved,
               peak_host_rss_gib=resource.getrusage(
                   resource.RUSAGE_SELF
               ).ru_maxrss / 2**20,
               n_prefill=len(prefills), n_decode=len(decodes), **extra)

    line = (f"{rec['model']:<8} {rec['datalen']:>6} {rec['method']:<13} "
            f"b{rec['sparse_budget']:<5} prefill {rec['prefill_s']:7.3f} s   "
            f"decode p50/mean/p95 {rec['decode_ms']:7.2f}/"
            f"{rec['decode_mean_ms']:7.2f}/{rec['decode_p95_ms']:7.2f} ms   "
            f"peak alloc/reserved {rec['peak_gib']:5.2f}/"
            f"{rec['peak_reserved_gib']:5.2f} GiB   "
            f"host RSS {rec['peak_host_rss_gib']:5.2f} GiB"
            + (f"  [{variant}]" if variant else ""))
    if "group_union_block_frac" in rec:
        line += (f"   blocks/query {rec['per_query_block_frac']*100:5.2f}%"
                 f"   GQA-union {rec['group_union_block_frac']*100:5.2f}%"
                 + ("" if args.real_text else "  [RANDOM TEXT -- not the traffic number]"))
    print(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
