#!/usr/bin/env python
"""
Preflight for a ShadowKV / streaming Quest campaign. Prints [OK] / [MISSING] per item and
refuses to return 0 while anything is missing, so a launcher can gate on it.

  source repro/shadowkv/env_m1.sh
  $PY repro/shadowkv/preflight.py --model $SHADOWKV_LLAMA32_PATH --template llama-3 \
      --datalen 8192 --method quest_streaming --sparse_budget 1024

It does not run a cell. A one-cell smoke on the same machine is still required
before opening a pool -- preflight prints that reminder last.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHADOWKV = os.path.join(REPO, "ShadowKV")

RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]

MISSING = []


def ok(label, detail=""):
    print(f"[OK]      {label}" + (f"  --  {detail}" if detail else ""))


def missing(label, detail=""):
    MISSING.append(label)
    print(f"[MISSING] {label}" + (f"  --  {detail}" if detail else ""))


def check_git():
    try:
        sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain"], text=True)
        untracked = [l[3:] for l in dirty.splitlines() if l.startswith("??")]
        ok("git", f"{sha}{' +dirty' if dirty.strip() else ''}")
        # An untracked .py that something imports breaks every other machine on pull.
        risky = [f for f in untracked if f.endswith(".py")]
        if risky:
            print(f"          note: untracked .py files present: {risky[:5]}")
    except Exception as e:
        missing("git", str(e))


def check_env():
    print(f"          python: {sys.executable}")
    try:
        import torch, transformers, flash_attn
        ok("env", f"torch {torch.__version__} | transformers {transformers.__version__} | flash_attn {flash_attn.__version__}")
        if not torch.cuda.is_available():
            missing("cuda", "torch.cuda.is_available() is False")
        else:
            ok("cuda", f"{torch.cuda.device_count()} device(s), driver ordering {os.environ.get('CUDA_DEVICE_ORDER', 'UNSET')}")
            if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
                missing("CUDA_DEVICE_ORDER", "must be PCI_BUS_ID so nvidia-smi and CUDA agree on GPU numbers")
    except Exception as e:
        missing("env", str(e))


def check_kernels():
    sys.path.insert(0, SHADOWKV)
    try:
        from kernels import shadowkv  # noqa: F401
        so = glob.glob(os.path.join(SHADOWKV, "kernels", "*.so"))
        ok("shadowkv CUDA extension", os.path.basename(so[0]) if so else "imported")
    except Exception as e:
        missing("shadowkv CUDA extension", f"{e}  (rebuild: python setup.py build_ext --inplace)")


def check_model(model_path):
    if not os.path.isdir(model_path):
        missing("model dir", model_path)
        return None
    incomplete = glob.glob(os.path.join(model_path, "**", "*.incomplete"), recursive=True)
    if incomplete:
        missing("model blobs", f"{len(incomplete)} *.incomplete file(s) -- a partial download hangs silently")
        return None
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        missing("model config", cfg_path)
        return None
    cfg = json.load(open(cfg_path))
    # every shard the index names must resolve, symlinks included
    idx = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        shards = set(json.load(open(idx))["weight_map"].values())
        for s in sorted(shards):
            f = os.path.join(model_path, s)
            if not os.path.exists(f) or os.path.getsize(f) == 0:
                missing("model shard", f)
                return None
    ok("model", f"{cfg.get('model_type')} | {cfg.get('num_hidden_layers')}L | "
                f"heads {cfg.get('num_attention_heads')}/{cfg.get('num_key_value_heads')} | "
                f"head_dim {cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads'])}")
    return cfg


def check_dispatch(model_path):
    sys.path.insert(0, SHADOWKV)
    try:
        from models import choose_model_class
        cls = choose_model_class(model_path)
        ok("model class", cls.__name__)
        return cls
    except Exception as e:
        missing("model class", str(e))
        return None


def check_dataset(template, datalen, tasks):
    root = os.path.join(SHADOWKV, "data", "ruler", "data", template, str(datalen))
    found, empty = [], []
    for t in tasks:
        f = os.path.join(root, t, "validation.jsonl")
        if os.path.isfile(f) and os.path.getsize(f) > 0:
            found.append((t, sum(1 for _ in open(f))))
        else:
            empty.append(t)
    if empty:
        missing("ruler data", f"{template}/{datalen}: absent {empty}")
    if found:
        counts = {n for _, n in found}
        ok("ruler data", f"{template}/{datalen}: {len(found)} task(s), "
                         f"{'all ' + str(counts.pop()) + ' lines' if len(counts) == 1 else 'line counts differ: ' + str(dict(found))}")


def check_method(method, datalen, sparse_budget, page_size, chunk_size, cfg):
    method = method.lower()
    if method not in (
        "full", "quest_streaming", "shadowkv", "shadowkv_cpu",
        "exact_block_lse_streaming", "exact_block_max_streaming",
        "exact_block_lse_softmax_streaming",
        "exact_block_max_softmax_streaming",
        "pariskv_official",
        "pariskv_author_common",
        "retroinfer_reference_streaming",
        "infllm_author_common",
        "magicpig_author_common",
        "pqcache_author_common",
        "pqcache_author_native",
        "adaptive_centroid_lse_streaming",
        "adaptive_centroid_lse_streaming_prefix4",
        "adaptive_centroid_lse_streaming_prefix4_querymean",
    ):
        missing("method", f"unknown '{method}'")
        return
    if method == "full":
        ok("method", "full attention -- budget flags ignored")
        return
    if method == "pariskv_author_common":
        author_root = os.environ.get("PARISKV_AUTHOR_ROOT", "")
        if not os.path.isdir(os.path.join(author_root, ".git")):
            missing("ParisKV author checkout", author_root or "PARISKV_AUTHOR_ROOT is unset")
        else:
            ok("ParisKV author checkout", author_root)
    if method == "infllm_author_common":
        author_root = os.environ.get("INFLLM_AUTHOR_ROOT", "")
        context = os.path.join(
            author_root, "inf_llm", "attention", "context_manager.py"
        )
        if not os.path.isfile(context):
            missing("InfLLM author checkout", author_root or "INFLLM_AUTHOR_ROOT is unset")
        else:
            ok("InfLLM author checkout", author_root)
    if method == "magicpig_author_common":
        author_root = os.environ.get("MAGICPIG_AUTHOR_ROOT", "")
        server = os.path.join(author_root, "models", "attnserver.py")
        if not os.path.isfile(server):
            missing("MagicPIG author checkout", author_root or "MAGICPIG_AUTHOR_ROOT is unset")
        else:
            ok("MagicPIG author checkout", author_root)
        cpu_flags = ""
        try:
            cpu_flags = open("/proc/cpuinfo").read().lower()
        except OSError:
            pass
        if "avx512f" not in cpu_flags:
            missing("MagicPIG CPU ISA", "released kernels hard-code -mavx512f")
        else:
            try:
                import lsh  # noqa: F401
                import sparse_attention_cpu  # noqa: F401
                ok("MagicPIG native kernels", "lsh + sparse_attention_cpu")
            except Exception as exc:
                missing("MagicPIG native kernels", str(exc))
    if method in ("pqcache_author_common", "pqcache_author_native"):
        author_root = os.environ.get("PQCACHE_AUTHOR_ROOT", "")
        search = os.path.join(
            author_root, "vq_method", "retrieval_based", "pq_search.py"
        )
        if not os.path.isfile(search):
            missing("PQCache author checkout", author_root or "PQCACHE_AUTHOR_ROOT is unset")
        else:
            ok("PQCache author checkout", author_root)
        try:
            import kmeans_gpu  # noqa: F401
            ok("PQCache kmeans component", "kmeans-gpu")
        except Exception as exc:
            missing("PQCache kmeans component", str(exc))
        if method == "pqcache_author_native":
            lfu = glob.glob(os.path.join(
                author_root, "vq_method", "retrieval_based", "lfu", "build",
                "lfucache*.so",
            ))
            if not lfu:
                missing("PQCache native LFU", "run setup_upstream_kv_baselines.sh")
            else:
                try:
                    from pathlib import Path
                    from models.pqcache_author_native_cache import (
                        PQCacheAuthorNativeCache,
                    )
                    bridge = object.__new__(PQCacheAuthorNativeCache)
                    bridge.author_root = Path(author_root).resolve()
                    runtime = bridge._import_author_runtime()
                    ok("PQCache native runtime", runtime.__file__)
                except Exception as exc:
                    missing("PQCache native runtime", str(exc))
    if sparse_budget >= datalen:
        missing("budget", f"sparse_budget {sparse_budget} must be < datalen {datalen}")
    if method == "quest_streaming":
        if sparse_budget % page_size:
            missing("budget", f"{sparse_budget} is not a multiple of page_size {page_size}")
        else:
            ok("method", f"{method} | budget {sparse_budget} | page {page_size} | {sparse_budget // page_size} pages selected")
    else:
        if sparse_budget % chunk_size:
            missing("budget", f"{sparse_budget} is not a multiple of chunk_size {chunk_size}")
        else:
            ok("method", f"{method} | budget {sparse_budget} | chunk {chunk_size}")


def check_memory(cfg, datalen, method):
    """A dense KV cache at this length must fit next to the weights."""
    if cfg is None:
        return
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    kv_bytes = (cfg["num_hidden_layers"] * cfg["num_key_value_heads"]
                * (datalen + 2048) * head_dim * 2 * 2)  # k+v, bf16
    params = cfg["hidden_size"] * cfg["vocab_size"] * 2  # rough; weights dominate
    try:
        import torch
        total = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return
    # streaming Quest and full both hold the whole cache; ShadowKV holds V plus buffers
    factor = 1.0 if method.lower() in (
        "full", "quest_streaming", "exact_block_lse_streaming",
        "exact_block_max_streaming", "exact_block_lse_softmax_streaming",
        "exact_block_max_softmax_streaming",
        "pariskv_official",
        "pariskv_author_common",
        "retroinfer_reference_streaming",
        "infllm_author_common",
        "magicpig_author_common",
        "pqcache_author_common",
        "pqcache_author_native",
        "adaptive_centroid_lse_streaming",
        "adaptive_centroid_lse_streaming_prefix4",
        "adaptive_centroid_lse_streaming_prefix4_querymean",
    ) else 0.6
    est = kv_bytes * factor
    print(f"          KV cache at {datalen} ~ {est / 2**30:.1f} GiB on a {total / 2**30:.0f} GiB card "
          f"(weights not counted; measure before trusting)")


def check_disk_and_gpu(results_root, gpus):
    parent = results_root
    while parent and not os.path.isdir(parent):
        parent = os.path.dirname(parent)
    free = shutil.disk_usage(parent).free
    ok("disk", f"{free / 2**30:.0f} GiB free at {parent}")

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader"], text=True)
        busy = [l for l in out.strip().splitlines() if l.strip()]
        ok("gpu", f"{len(busy)} running compute process(es)" + (f": {busy}" if busy else ""))
    except Exception as e:
        missing("gpu", str(e))
    if gpus:
        policy = os.environ.get("SHADOWKV_GPU_POLICY", "see the machine launcher")
        print(f"          pool roster: {gpus}  ({policy})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", required=True, choices=["llama-3", "qwen"])
    ap.add_argument("--datalen", type=int, required=True)
    ap.add_argument("--tasks", default=",".join(RULER_TASKS))
    ap.add_argument("--method", default="full")
    ap.add_argument("--sparse_budget", type=int, default=1024)
    ap.add_argument("--page_size", type=int, default=16)
    ap.add_argument("--chunk_size", type=int, default=8)
    args = ap.parse_args()

    print(f"=== preflight: {args.method} @ {args.datalen} on {os.path.basename(args.model.rstrip('/'))} ===")
    check_git()
    check_env()
    check_kernels()
    cfg = check_model(args.model)
    check_dispatch(args.model)
    check_dataset(args.template, args.datalen, [t for t in args.tasks.split(",") if t])
    check_method(args.method, args.datalen, args.sparse_budget, args.page_size, args.chunk_size, cfg)
    check_memory(cfg, args.datalen, args.method)
    check_disk_and_gpu(os.environ.get("SHADOWKV_RESULTS_ROOT", "/tmp"),
                       os.environ.get("SHADOWKV_POOL_GPUS", ""))

    print()
    if MISSING:
        print(f"*** {len(MISSING)} ITEM(S) MISSING -- do not launch: {MISSING}")
        return 1
    print("all checks passed. Next: run ONE cell at --num_samples 2 on this machine "
          "and read a real number out of it before opening a pool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
