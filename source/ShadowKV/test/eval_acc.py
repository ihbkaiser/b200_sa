################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

# OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 8 test/eval_acc.py --datalen 131072 --method shadowKV --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multikey_3,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/cwe,ruler/fwe,ruler/qa_1,ruler/qa_2" --sparse_budget 896 --rank 160 --chunk_size 8

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import warnings
warnings.filterwarnings("ignore")

import torch
import gc
from termcolor import colored
from argparse import ArgumentParser, Namespace

import torch.distributed as dist
import datetime

class DistConfig:
    def __init__(self, is_distributed, rank, world_size, device, master_process):
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.master_process = master_process

def init_dist():
    rank = int(os.environ.get("RANK", -1))
    is_distributed = rank != -1
    if is_distributed:
        dist.init_process_group(backend="nccl",timeout=datetime.timedelta(seconds=60*90))
        world_size = int(os.environ["WORLD_SIZE"])

        device = f"cuda:{rank}" 
        torch.cuda.set_device(device)
        master_process = (
            rank == 0
        )
    else:
        device = "cuda:0"
        world_size = 1
        master_process = True

    if master_process:
        print(colored(f"[Dist init] world_size={world_size}", 'cyan'))
    
    return DistConfig(is_distributed, rank, world_size, device, master_process)

def parse_args() -> Namespace:
    def str_to_list(arg):
        return arg.split(',')
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="gradientai/Llama-3-8B-Instruct-Gradient-1048k")
    p.add_argument("--dataset_name", type=str_to_list, default=["ruler/niah_single_1"])
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--sample_start", type=int, default=0,
                   help="first dataset row for an independent non-overlapping shard")
    p.add_argument("--sample_stop", type=int, default=-1,
                   help="exclusive dataset row; -1 means the end")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--datalen", type=int, default=128*1024, help="The length of the context.")
    p.add_argument("--max_new_tokens", type=int, default=-1,
                   help="override the benchmark generation cap; -1 uses its official local value")
    p.add_argument("--generation_temperature", type=float, default=0.0)
    p.add_argument("--generation_top_p", type=float, default=1.0)
    p.add_argument("--generation_top_k", type=int, default=-1,
                   help="-1 disables top-k filtering, matching the KVPress AIME runner")
    p.add_argument("--generation_seed", type=int, default=-1,
                   help="sampling seed reset for each independent benchmark row")
    p.add_argument("--method", type=str, default="full")
    p.add_argument("--sparse_budget", type=int, default=2048)
    p.add_argument("--rank", type=int, default=160)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--minference", action='store_true', default=False)
    # Streaming-Quest knobs. They are printed by print_kv_stats() and encoded
    # in the output filename so a cell can never be read back without them.
    p.add_argument("--page_size", type=int, default=16, help="Quest page size")
    p.add_argument("--dense_layers", type=int, default=2, help="Quest: leading layers kept dense")
    p.add_argument("--group_reduce", type=str, default='max', choices=['max', 'sum'],
                   help="Quest: how per-query-head page scores are reduced across a GQA group")
    p.add_argument("--quest_prefix_tokens", type=int, default=None,
                   help="Quest: exact prefix tokens added outside sparse_budget")
    p.add_argument("--streaming_update_interval", type=lambda v: None if int(v) <= 0 else int(v),
                   default=None,
                   help="batch-flush cadence in tokens; the exact suffix is the local "
                        "window plus a 0..interval buffer. Defaults to the block size, "
                        "which is the per-block lifecycle this harness shipped with.")
    p.add_argument("--streaming_recent_tokens", type=int, default=32,
                   help="streaming variants: rolling exact suffix outside sparse_budget")
    p.add_argument("--streaming_offload", action="store_true",
                   help="keep exact streaming K/V in pinned CPU memory")
    p.add_argument("--streaming_gather_backend", choices=["auto", "uva", "torch"],
                   default="auto", help="CPU-offload gather implementation")
    p.add_argument("--streaming_router_backend", choices=["torch", "triton"],
                   default="torch", help="adaptive-centroid decode router")
    p.add_argument("--streaming_refine_factor", type=float, default=1.0,
                   help="adaptive router candidate multiplier for live-query exact-LSE reranking")
    p.add_argument("--streaming_refine_candidate_ratio", type=float, default=None,
                   help="adaptive router candidate fraction for live-query exact-LSE reranking")
    p.add_argument("--streaming_refine_tokens", action="store_true",
                   help="select individual candidate tokens after exact reranking")
    p.add_argument("--streaming_max_components", type=int, default=None,
                   help="maximum centers a difficult adaptive block may receive")
    compact = p.add_mutually_exclusive_group()
    compact.add_argument(
        "--streaming_compact_metadata", dest="streaming_compact_metadata",
        action="store_true", help="store only active adaptive-centroid components",
    )
    compact.add_argument(
        "--no_streaming_compact_metadata", dest="streaming_compact_metadata",
        action="store_false", help="store a dense block-by-component tensor",
    )
    p.set_defaults(streaming_compact_metadata=None)
    p.add_argument("--streaming_center_bits", type=int, choices=[4, 8, 16], default=None,
                   help="per-center symmetric quantization precision")
    p.add_argument("--query_robust_vertices_path", type=str, default=None,
                   help="Qwen3 Query-Robust BF16 vertex asset")
    p.add_argument("--query_robust_model_fingerprint", type=str, default=None,
                   help="checkpoint fingerprint expected by the QR asset")
    p.add_argument("--query_robust_vertices_sha256", type=str, default=None,
                   help="SHA-256 expected for the QR vertex asset")
    p.add_argument("--query_robust_num_vertices", type=int, default=32)
    p.add_argument("--query_robust_solver_iters", type=int, default=24)
    p.add_argument("--query_robust_solver_lr", type=float, default=0.25)
    p.add_argument("--query_robust_score_alpha", type=float, default=1.0)
    p.add_argument("--query_robust_uniform_p", action="store_true")
    p.add_argument("--query_robust_summary_page_batch", type=int, default=None,
                   help="number of pages summarized per PyTorch QR build batch")
    p.add_argument("--retroinfer_prefix_tokens", type=int, default=4,
                   help="RetroInfer reference: exact leading tokens")
    p.add_argument("--retroinfer_recent_tokens", type=int, default=64,
                   help="RetroInfer reference: minimum exact rolling suffix")
    p.add_argument("--retroinfer_update_segment", type=int, default=1024,
                   help="RetroInfer reference: tokens per appended wave-index update")
    p.add_argument("--retroinfer_average_cluster_size", type=int, default=16,
                   help="RetroInfer reference: target tokens per cluster")
    p.add_argument("--retroinfer_estimation_ratio", type=float, default=0.232,
                   help="RetroInfer reference: fraction of clusters in estimation zone")
    p.add_argument("--retroinfer_kmeans_iters", type=int, default=10,
                   help="RetroInfer reference: spherical k-means iterations")
    p.add_argument("--retroinfer_author_root", type=str,
                   default=os.environ.get("RETROINFER_AUTHOR_ROOT"),
                   help="RetrievalAttention checkout for retroinfer_author_common")
    p.add_argument("--retroinfer_n_centroids", type=int, default=0,
                   help="RetroInfer: fixed prefill cluster count (0 = derive "
                        "from --retroinfer_average_cluster_size)")
    p.add_argument("--retroinfer_n_segment", type=int, default=16,
                   help="RetroInfer: k-means training segments at prefill")
    p.add_argument("--pariskv_author_root", type=str, default=None,
                   help="ParisKV checkout used by the common-forward author router")
    p.add_argument("--quill_router_exact_fraction", type=float, default=0.125,
                   help="ShadowKV-QUILL: fraction of per-pool tokens retained as exact routing landmarks")
    p.add_argument("--quill_router_score_chunk", type=int, default=1024,
                   help="ShadowKV-QUILL: independent token pool used by canonical QUILL ranking")
    p.add_argument("--router_aggregation", choices=["max", "logsumexp"], default="max",
                   help="exact-key aggregation; logsumexp requires exact_fraction=1")
    p.add_argument("--router_centroids", type=int, default=2,
                   help="symmetric centroid-LSE components per block")
    p.add_argument(
        "--router_centroid_method",
        choices=[
            "farthest_body",
            "kmeans",
            "minimax2",
            "minimax2_sse",
            "scatter2",
            "cosine2",
            "self_lse2",
            "self_lse2_peak",
            "self_lse2_iso",
            "self_lse_adaptive_iso",
            "direct_lse2",
        ],
        default="kmeans",
    )
    p.add_argument("--router_split_fraction", type=float, default=None,
                   help="mean number of extra centers per block for the adaptive 1..S allocator")
    p.add_argument("--router_calibration_tokens", type=int, default=8,
                   help="direct_lse2: final prompt queries used for context-local LSE calibration")
    p.add_argument("--router_self_lse_temperatures", type=str, default="1",
                   help="comma-separated normalized-key scales for self_lse2")
    p.add_argument("--shadow_outlier_chunks", type=int, default=48,
                   help="ShadowKV: number of permanently resident geometric-outlier blocks per layer/head")
    p.add_argument("--out_root", type=str, default='archive', help="where per-cell jsonl is written")
    # M51 (routed multi-anchor PQ LSE retrieval)
    p.add_argument("--m51_anchors", type=str, default=None,
                   help="npz from repro/shadowkv/m51_fit_anchors.py (offline anchors + calib queries)")
    p.add_argument("--m51_leaf", type=int, default=8)
    p.add_argument("--m51_anchors_n", type=int, default=8)
    p.add_argument("--m51_coverage", type=float, default=0.90)
    p.add_argument("--m51_eta_q", type=float, default=0.999)
    p.add_argument("--m51_pq_mode", choices=["offline", "per_context", "warm"], default="offline",
                   help="offline reuses a codebook fitted on calibration documents; "
                        "per_context refits it every prompt, as the probe does")
    p.add_argument("--m51_pq_warm_iters", type=int, default=2,
                   help="Lloyd steps for pq_mode=warm, seeded from the offline codebook")
    p.add_argument("--m51_pq_codes", type=int, default=64,
                   help="PQ codewords per subquantiser. Quest's page bound is EXACT given "
                        "min/max; M51 quantises its tangent, and on retrieval among near-identical "
                        "distractors that quantisation is what loses. 256 costs 8 bits per "
                        "subquantiser instead of 6: 5.1%% of KV bytes against Quest's 6.25%%.")
    p.add_argument("--m51_group_select", choices=["shared", "per_query"], default="per_query",
                   help="shared ranks once per KV head so a GQA group's loads are one prefix")
    p.add_argument("--m51_decode_mode", choices=["gather", "dense_mask"], default="dense_mask",
                   help="dense_mask is the reference path: same output, reads the whole cache")
    p.add_argument("--m51_load_batch", type=int, default=128)
    p.add_argument("--m51_rank_anchors", type=int, default=1,
                   help="blocks are scored by the MIN over this many nearest anchors; "
                        "0 = every anchor. Each anchor gives a valid bound, so more is tighter")
    p.add_argument("--m51_rank_reduce", choices=["max", "softmax"], default="max",
                   help="how a GQA group's query heads are combined before the top-k")
    p.add_argument("--m51_select_mode", choices=["adaptive", "fixed"], default="adaptive",
                   help="fixed: rank once per KV head on the group-reduced bound and take "
                        "--sparse_budget tokens, so the budget is the traffic (Quest's shape)")
    p.add_argument("--traffic_out", type=str, default=None,
                   help="write M51's measured block traffic here; for M51 this is the "
                        "decode cost that matters, not wall clock")
    p.add_argument("--runtime_out", type=str, default=None,
                   help="measure prefill and per-step decode latency on one "
                        "prompt and write JSON here instead of scoring")
    p.add_argument("--runtime_steps", type=int, default=64,
                   help="timed decode steps (after --runtime_warmup)")
    p.add_argument("--runtime_warmup", type=int, default=8,
                   help="decode steps discarded before timing starts")
    p.add_argument("--allocation_out", type=str, default=None,
                   help="append the final prompt's adaptive-centroid budget audit as JSONL")
    p.add_argument("--cell_name", type=str, default=None,
                   help="exact output stem. Pass this from repro/shadowkv/cell_key.py so the "
                        "jsonl, the stamp and the pool marker all carry ONE name; without it "
                        "the stem is derived here and the two definitions can drift apart.")

    return p.parse_args()


def record_profile(llm, token, args):
    """Localize a slow decode step: GPU-bound, or launch-bound?

    The decisive number is the ratio of summed CUDA kernel time to wall time
    over the same steps.  Near 1 means the GPU is the bottleneck and the fix
    is arithmetic; far below 1 means the step is spent launching work, and
    the fix is fusing kernels rather than changing the method.
    """
    import time

    from torch.profiler import ProfilerActivity, profile

    steps = 8
    torch.cuda.synchronize()
    started = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(steps):
            llm.inference(input_ids=token, position_ids=llm.get_ctx(token))
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0 / steps

    events = prof.key_averages()
    cuda_us = sum(
        getattr(event, "self_device_time_total", 0) or 0 for event in events
    )
    launches = sum(
        event.count for event in events
        if (getattr(event, "self_device_time_total", 0) or 0) > 0
    )
    print("\n===== decode-step profile =====", flush=True)
    print(f"method                {args.method}")
    print(f"wall per step         {wall_ms:9.3f} ms  (profiler adds overhead)")
    print(f"CUDA kernel per step  {cuda_us / 1000.0 / steps:9.3f} ms")
    print(f"GPU busy fraction     {cuda_us / 1000.0 / steps / wall_ms:9.3f}")
    print(f"kernel launches/step  {launches / steps:9.1f}")
    print(events.table(
        sort_by="self_cuda_time_total", row_limit=12, max_name_column_width=55
    ), flush=True)
    print(events.table(
        sort_by="count", row_limit=12, max_name_column_width=55
    ), flush=True)
    print("===== end profile =====\n", flush=True)


def _gpu_state():
    """Temperature and SM clock of the card this process is pinned to."""
    import subprocess

    index = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--format=csv,noheader,nounits",
             "--query-gpu=temperature.gpu,clocks.sm"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        temperature, clock = (int(part) for part in out.split(","))
        return temperature, clock
    except Exception:
        return -1, -1


def _cool_to(target, timeout_s):
    """Wait for the card to cool before timing anything.

    Cells run back to back for an hour, and an A5000 that has been busy since
    the first cell is 30 C hotter by the sixth.  Measured on this machine, the
    same ShadowKV cell reads 57.8 ms on a rested card and 69.3 ms when it runs
    sixth -- a 20% swing decided by position in the queue, which is exactly the
    kind of difference a runtime table is supposed to be measuring.
    """
    import time

    if target <= 0:
        return _gpu_state()
    started = time.time()
    while True:
        temperature, clock = _gpu_state()
        if temperature < 0 or temperature <= target:
            return temperature, clock
        if time.time() - started > timeout_s:
            print(f"[runtime] cooldown timed out at {temperature} C", flush=True)
            return temperature, clock
        time.sleep(10)


def measure_runtime(llm, args, dataset_name, datalen):
    """Time one prompt through the same model object the accuracy runs use.

    Prefill and decode are reported separately because they stress different
    things: prefill is dense attention plus whatever index the method builds,
    decode is the per-step router and the KV fetch.  Steps are timed one at a
    time with a synchronize on each side, so the number is a real per-token
    latency rather than a throughput average that hides a slow tail.
    """
    import json
    import statistics
    import time

    from data.dataset import Dataset
    from models.tensor_op import sample_token

    dataset = Dataset(
        dataset_name, llm.tokenizer, datalen, 1, 0, 1
    )
    prompt = dataset.tokenized_prompts[0].to(llm.device)
    torch.cuda.reset_peak_memory_stats(llm.device)
    cool_c, cool_clock = _cool_to(
        int(os.environ.get("SHADOWKV_RUNTIME_COOL_C", "55")),
        int(os.environ.get("SHADOWKV_RUNTIME_COOL_TIMEOUT", "900")),
    )

    torch.cuda.synchronize()
    started = time.perf_counter()
    logits = llm.prefill(prompt)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started

    token = sample_token(logits[:, -1, :], temperature=0.0, top_p=1.0, top_k=1)
    llm.kv_cache.H2D()

    steps = []
    samples = []
    for index in range(args.runtime_warmup + args.runtime_steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        logits = llm.inference(
            input_ids=token, position_ids=llm.get_ctx(token)
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if index >= args.runtime_warmup:
            steps.append(elapsed * 1000.0)
        token = sample_token(
            logits[:, -1, :], temperature=0.0, top_p=1.0, top_k=1
        )
        if index % 64 == 0:
            samples.append(_gpu_state())

    samples.append(_gpu_state())
    if os.environ.get("SHADOWKV_RUNTIME_PROFILE") == "1":
        record_profile(llm, token, args)

    steps.sort()
    record = {
        "method": args.method,
        "dataset": dataset_name,
        "datalen": int(datalen),
        "prompt_tokens": int(prompt.shape[-1]),
        "sparse_budget": int(args.sparse_budget),
        "chunk_size": int(args.chunk_size),
        "offload": bool(args.streaming_offload),
        "gather_reuse": os.environ.get("STREAMING_GATHER_REUSE", "1") == "1",
        "prefill_seconds": round(prefill_seconds, 4),
        "prefill_tokens_per_second": round(
            prompt.shape[-1] / prefill_seconds, 1
        ),
        "decode_steps": len(steps),
        "decode_ms_mean": round(statistics.fmean(steps), 4),
        "decode_ms_median": round(statistics.median(steps), 4),
        "decode_ms_p90": round(steps[int(0.9 * (len(steps) - 1))], 4),
        "decode_tokens_per_second": round(
            1000.0 / statistics.fmean(steps), 2
        ),
        "peak_gpu_gib": round(
            torch.cuda.max_memory_allocated(llm.device) / 2**30, 3
        ),
        "attended_tokens": int(
            getattr(llm.kv_cache, "last_attention_tokens", 0) or 0
        ),
        # Thermal context.  A number without it is not comparable to a number
        # measured at a different point in a queue.
        "start_temperature_c": cool_c,
        "start_clock_mhz": cool_clock,
        "max_temperature_c": max((t for t, _ in samples), default=-1),
        "min_clock_mhz": min((c for _, c in samples if c > 0), default=-1),
        "mean_clock_mhz": round(
            sum(c for _, c in samples if c > 0)
            / max(1, sum(1 for _, c in samples if c > 0))
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.runtime_out)), exist_ok=True)
    with open(args.runtime_out, "w") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':

    args = parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    dataset_names = args.dataset_name
    num_samples = args.num_samples
    datalen = args.datalen
    sparse_budget = args.sparse_budget
    dtype = torch.bfloat16
    rank = args.rank
    chunk_size = args.chunk_size
    minference = args.minference
    method_key = args.method.lower()
    querymean_default = method_key.endswith("_querymean")
    if args.router_split_fraction is None:
        # Query-mean routing chooses 1..S centers independently per block by
        # fixed-price rate--distortion, rather than enforcing a mean quota.
        args.router_split_fraction = 0.0 if querymean_default else 1.0
    if args.streaming_compact_metadata is None:
        args.streaming_compact_metadata = querymean_default
    if args.streaming_center_bits is None:
        args.streaming_center_bits = 8 if querymean_default else 16
    if args.quest_prefix_tokens is None:
        args.quest_prefix_tokens = (
            4 * chunk_size
            if method_key.startswith("adaptive_centroid_lse_streaming")
            and "_prefix4" in method_key
            else 0
        )

    dist_config = init_dist()
    
    from evaluator import Evaluator
    from models import choose_model_class
    from data.dataset import Dataset
    
    evaluator = Evaluator(dist_config)
    
    if dist_config.master_process:
        print(colored(f"data_names: {dataset_names}", 'cyan'))
    
    LLM = choose_model_class(model_name)

    if args.max_new_tokens > 0:
        generation_reserve = args.max_new_tokens
    else:
        # Match the per-row defaults in NVIDIA/kvpress@71640b4. GPQA is the
        # one local extension: current upstream KVPress no longer registers it.
        reserve_by_dataset = []
        for name in dataset_names:
            if name == 'aime25':
                reserve_by_dataset.append(32000)
            elif name.startswith('gpqa'):
                reserve_by_dataset.append(16384)
            elif name == 'math500':
                reserve_by_dataset.append(4096)
            elif name == 'longbench-v2':
                reserve_by_dataset.append(16)
            else:
                reserve_by_dataset.append(2048)
        generation_reserve = max(reserve_by_dataset)
    llm_kwargs = dict(model_name=model_name, batch_size=batch_size, device=dist_config.device,
                      max_length=datalen+max(2048, generation_reserve), attn_mode=args.method, dtype=dtype,
                      sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size,
                      minference=minference)
    if args.method.lower() in {
        'quest_streaming',
        'query_robust', 'qr',
        'exact_block_lse_streaming', 'exact_block_max_streaming',
        'exact_block_lse_softmax_streaming',
        'exact_block_max_softmax_streaming',
    }:
        llm_kwargs.update(page_size=args.page_size, dense_layers=args.dense_layers,
                          group_reduce=args.group_reduce,
                          quest_prefix_tokens=args.quest_prefix_tokens,
                          quest_recent_tokens=args.streaming_recent_tokens,
                          streaming_update_interval=args.streaming_update_interval)
    if args.method.lower() in {'quest_streaming', 'query_robust', 'qr'}:
        llm_kwargs.update(
            streaming_offload=args.streaming_offload,
            streaming_gather_backend=args.streaming_gather_backend,
        )
    if args.method.lower() in {'query_robust', 'qr'}:
        llm_kwargs.update(
            query_robust_vertices_path=args.query_robust_vertices_path,
            query_robust_model_fingerprint=args.query_robust_model_fingerprint,
            query_robust_vertices_sha256=args.query_robust_vertices_sha256,
            query_robust_num_vertices=args.query_robust_num_vertices,
            query_robust_solver_iters=args.query_robust_solver_iters,
            query_robust_solver_lr=args.query_robust_solver_lr,
            query_robust_score_alpha=args.query_robust_score_alpha,
            query_robust_uniform_p=args.query_robust_uniform_p,
            query_robust_summary_page_batch=args.query_robust_summary_page_batch,
        )
    if args.method.lower().startswith('exact_block_'):
        # Exact controls share the streaming refinement implementation with
        # the adaptive router.  Keep these arguments outside the adaptive-only
        # branch below; otherwise a cell named ``tokref2x`` silently executes
        # the one-shot x1 path.
        llm_kwargs.update(
            streaming_refine_factor=args.streaming_refine_factor,
            streaming_refine_tokens=args.streaming_refine_tokens,
            streaming_offload=args.streaming_offload,
            streaming_gather_backend=args.streaming_gather_backend,
        )
    if args.method.lower() == 'retroinfer_reference_streaming':
        llm_kwargs.update(
            retroinfer_prefix_tokens=args.retroinfer_prefix_tokens,
            retroinfer_recent_tokens=args.retroinfer_recent_tokens,
            retroinfer_update_segment=args.retroinfer_update_segment,
            retroinfer_average_cluster_size=args.retroinfer_average_cluster_size,
            retroinfer_estimation_ratio=args.retroinfer_estimation_ratio,
            retroinfer_kmeans_iters=args.retroinfer_kmeans_iters,
        )
    if args.method.lower() == 'pqcache_author_common':
        # PQCache's store is the one thing about it that is a deployment
        # choice rather than an algorithm, so it follows the same residency
        # flag as every other retrieval method.
        llm_kwargs.update(streaming_offload=args.streaming_offload)
    if args.method.lower() == 'retroinfer_author_common':
        llm_kwargs.update(
            retroinfer_author_root=args.retroinfer_author_root,
            retroinfer_average_cluster_size=args.retroinfer_average_cluster_size,
            retroinfer_n_centroids=args.retroinfer_n_centroids,
            retroinfer_n_segment=args.retroinfer_n_segment,
            retroinfer_estimation_ratio=args.retroinfer_estimation_ratio,
            retroinfer_kmeans_iters=args.retroinfer_kmeans_iters,
            quest_prefix_tokens=args.quest_prefix_tokens,
            quest_recent_tokens=args.streaming_recent_tokens,
            streaming_update_interval=args.streaming_update_interval,
            streaming_offload=args.streaming_offload,
            streaming_gather_backend=args.streaming_gather_backend,
        )
    if args.method.lower() == 'pariskv_author_common':
        llm_kwargs.update(
            pariskv_author_root=args.pariskv_author_root,
            quest_prefix_tokens=args.quest_prefix_tokens,
            quest_recent_tokens=args.streaming_recent_tokens,
            streaming_update_interval=args.streaming_update_interval,
            streaming_offload=args.streaming_offload,
            streaming_gather_backend=args.streaming_gather_backend,
        )
    if args.method.lower() in {'shadowkv_quill', 'shadowkv_keydiff'}:
        llm_kwargs.update(
            quill_router_exact_fraction=args.quill_router_exact_fraction,
            quill_router_score_chunk=args.quill_router_score_chunk,
            router_aggregation=args.router_aggregation,
        )
    if args.method.lower() in {
        'shadowkv_centroid_lse', 'adaptive_centroid_lse',
        'adaptive_centroid_lse_prefix4',
        'adaptive_centroid_lse_streaming',
        'adaptive_centroid_lse_streaming_prefix4',
        'adaptive_centroid_lse_streaming_prefix4_querymean',
    }:
        llm_kwargs.update(
            router_centroids=args.router_centroids,
            router_centroid_method=args.router_centroid_method,
            router_split_fraction=args.router_split_fraction,
            router_calibration_tokens=args.router_calibration_tokens,
            self_lse_temperatures=tuple(
                float(item) for item in args.router_self_lse_temperatures.split(",") if item
            ),
        )
        if (
            args.method.lower().startswith('adaptive_centroid_lse_streaming')
            or args.method.lower().startswith('exact_block_')
        ):
            llm_kwargs.update(
                quest_prefix_tokens=args.quest_prefix_tokens,
                quest_recent_tokens=args.streaming_recent_tokens,
                streaming_update_interval=args.streaming_update_interval,
                group_reduce=args.group_reduce,
                streaming_offload=args.streaming_offload,
                streaming_gather_backend=args.streaming_gather_backend,
                streaming_router_backend=args.streaming_router_backend,
                streaming_refine_factor=args.streaming_refine_factor,
                streaming_refine_candidate_ratio=args.streaming_refine_candidate_ratio,
                streaming_refine_tokens=args.streaming_refine_tokens,
                streaming_max_components=args.streaming_max_components,
                streaming_compact_metadata=args.streaming_compact_metadata,
                streaming_center_bits=args.streaming_center_bits,
            )
    if args.method.lower() in {'shadowkv', 'shadowkv_cpu', 'shadowkv_quill', 'shadowkv_keydiff', 'shadowkv_centroid_lse'}:
        llm_kwargs.update(shadow_outlier_chunks=args.shadow_outlier_chunks)
    if args.method.lower() == 'm51':
        llm_kwargs.update(m51_anchors=args.m51_anchors, m51_leaf=args.m51_leaf,
                          m51_anchors_n=args.m51_anchors_n, m51_coverage=args.m51_coverage,
                          m51_eta_q=args.m51_eta_q, dense_layers=args.dense_layers,
                          m51_pq_mode=args.m51_pq_mode, m51_pq_warm_iters=args.m51_pq_warm_iters,
                          m51_pq_codes=args.m51_pq_codes, m51_group_select=args.m51_group_select,
                          m51_decode_mode=args.m51_decode_mode, m51_load_batch=args.m51_load_batch,
                          m51_select_mode=args.m51_select_mode, m51_budget=args.sparse_budget,
                          m51_rank_anchors=args.m51_rank_anchors,
                          m51_rank_reduce=args.m51_rank_reduce)

    llm = LLM(**llm_kwargs)

    if dist_config.master_process:
        llm.print_kv_stats()

    if args.runtime_out:
        measure_runtime(llm, args, dataset_names[0], datalen)
        sys.exit(0)

    for dataset_name in dataset_names:
        dataset = Dataset(dataset_name, llm.tokenizer, datalen, num_samples, evaluator.dist_config.rank, evaluator.dist_config.world_size)
        if args.max_new_tokens > 0:
            dataset.gen_len = args.max_new_tokens
        if args.sample_start or args.sample_stop >= 0:
            stop = None if args.sample_stop < 0 else args.sample_stop
            dataset.tokenized_prompts = dataset.tokenized_prompts[args.sample_start:stop]
            dataset.gt = dataset.gt[args.sample_start:stop]
            if dataset.classes is not None:
                dataset.classes = dataset.classes[args.sample_start:stop]
            if dataset.metadata is not None:
                dataset.metadata = dataset.metadata[args.sample_start:stop]
            dataset.num_samples = len(dataset.tokenized_prompts)
            if dataset.num_samples == 0:
                raise ValueError(
                    f"empty sample shard [{args.sample_start}:{args.sample_stop}]"
                )
        if args.cell_name:
            out_path = f"{args.out_root}/{args.cell_name}.jsonl"
        else:
            if args.method.lower() == 'm51':
                cell = (f"{dataset_name}_{datalen}_m51_c{args.m51_coverage}_l{args.m51_leaf}"
                        f"_a{args.m51_anchors_n}_q{args.m51_eta_q}_d{args.dense_layers}"
                        f"_{'v2' if (args.m51_pq_mode, args.m51_group_select, args.m51_decode_mode) == ('offline','per_query','dense_mask') else 'v1'}")
            elif args.method.lower() == 'full':
                cell = f"{dataset_name}_{datalen}_full"
            else:
                cell = f"{dataset_name}_{datalen}_{args.method}_b{sparse_budget}_r{rank}_c{chunk_size}"
            out_path = f"{args.out_root}/{model_name.rstrip('/').split('/')[-1]}/{cell}.jsonl"
        evaluator.test(
            llm, dataset, out_path, args.method,
            temperature=args.generation_temperature,
            top_p=args.generation_top_p,
            top_k=args.generation_top_k,
            sample_seed=(args.generation_seed if args.generation_seed >= 0 else None),
        )

        if args.allocation_out and (
            hasattr(llm.kv_cache, "router_component_count")
            or hasattr(llm.kv_cache, "component_count")
        ):
            import json as _json
            import os as _os

            histogram = {str(r): 0 for r in range(1, args.chunk_size + 1)}
            total_blocks = 0
            total_components = 0
            expected_components = 0
            layer_averages = []
            if hasattr(llm.kv_cache, "router_component_count"):
                count_tensors = llm.kv_cache.router_component_count
            else:
                count_tensors = []
                for layer_idx, state in enumerate(llm.kv_cache.block_state):
                    first, last = state.candidate_block_range
                    count_tensors.append(
                        llm.kv_cache.component_count[
                            layer_idx, :, :, first:last
                        ]
                    )
            for counts in count_tensors:
                if counts is None:
                    continue
                flat = counts.long().reshape(-1)
                bincount = torch.bincount(flat, minlength=args.chunk_size + 1)
                for r in range(1, args.chunk_size + 1):
                    histogram[str(r)] += int(bincount[r].item())
                total_blocks += flat.numel()
                total_components += int(flat.sum().item())
                blocks_per_group = counts.shape[-1]
                groups = counts.numel() // blocks_per_group
                expected_components += groups * (
                    blocks_per_group
                    + round(args.router_split_fraction * blocks_per_group)
                )
                layer_averages.append(float(flat.float().mean().item()))
            record = {
                "model": model_name.rstrip('/').split('/')[-1],
                "dataset": dataset_name,
                "datalen": datalen,
                "sample_slice": [args.sample_start, args.sample_stop],
                "total_blocks": total_blocks,
                "total_components": total_components,
                "expected_components": expected_components,
                "budget_exact": total_components == expected_components,
                "mean_components_per_block": (
                    total_components / total_blocks if total_blocks else None
                ),
                "max_components": max(
                    (int(r) for r, n in histogram.items() if n), default=None
                ),
                "histogram": histogram,
                "layer_mean_min": min(layer_averages) if layer_averages else None,
                "layer_mean_max": max(layer_averages) if layer_averages else None,
            }
            _os.makedirs(_os.path.dirname(args.allocation_out) or ".", exist_ok=True)
            with open(args.allocation_out, "a", encoding="utf-8") as handle:
                handle.write(_json.dumps(record) + "\n")
            print("[allocation audit]", _json.dumps(record))
    
    if args.traffic_out and hasattr(llm.kv_cache, "traffic"):
        import json as _json
        stats = llm.kv_cache.traffic()
        if stats:
            stats.update(model=model_name.rstrip('/').split('/')[-1], datalen=datalen,
                         method=args.method, coverage=args.m51_coverage,
                         leaf=args.m51_leaf, anchors=args.m51_anchors_n,
                         dataset=dataset_names[0])
            os.makedirs(os.path.dirname(args.traffic_out) or ".", exist_ok=True)
            with open(args.traffic_out, "a") as fh:
                fh.write(_json.dumps(stats) + "\n")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    evaluator.summarize()
