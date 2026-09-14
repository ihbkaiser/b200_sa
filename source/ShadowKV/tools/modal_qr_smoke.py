#!/usr/bin/env python3
"""Run a model-free Query-Robust CUDA smoke on a Modal B200.

This intentionally validates the vendored Qwen3 asset, the QR summary solver,
GQA routing, and exact gather contract. It does not load model weights or make
a RULER quality claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import modal


APP = modal.App("b200-sa-qwen3-query-robust-smoke")
SHADOWKV_ROOT = Path(__file__).resolve().parents[1]
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "numpy")
    .add_local_dir(
        SHADOWKV_ROOT,
        remote_path="/root/ShadowKV",
        ignore=["__pycache__", "*.pyc", "*.so", "build", "dist"],
    )
)


@APP.function(image=IMAGE, gpu="B200", timeout=900, include_source=True)
def qr_smoke() -> dict[str, object]:
    import sys

    import torch

    sys.path.insert(0, "/root/ShadowKV")
    from models.query_robust_cache import StreamingQueryRobustCache

    if not torch.cuda.is_available():
        raise RuntimeError("Modal QR smoke requires CUDA")
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability < (10, 0):
        raise RuntimeError(f"Expected B200/Blackwell, got compute capability {capability}")

    asset = Path(
        "/root/ShadowKV/artifacts/query_robust/qwen3_4b_128k/"
        "qwen3_4b_qr_vertices_m32_128k.pt"
    )
    config = SimpleNamespace(
        hidden_size=2560,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=36,
        head_dim=128,
    )
    cache = StreamingQueryRobustCache(
        config,
        max_length=256,
        device="cuda:0",
        dtype=torch.bfloat16,
        sparse_budget=32,
        block_size=8,
        dense_layers=0,
        prefix_tokens=0,
        recent_tokens=0,
        vertices_path=asset,
        model_id="Qwen3-4B-Instruct-2507",
        model_fingerprint="cdbee75f17c01a7cc42f958dc650907174af0554",
        vertices_sha256="189b839536e53dac532b032504311b803438d0a968a5aed1db90223f0e76fd68",
        num_vertices=32,
        solver_iters=24,
        solver_lr=0.25,
        score_alpha=1.0,
        summary_page_batch=8,
    )

    torch.manual_seed(20260914)
    keys = torch.randn(1, 8, 128, 128, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    for layer_idx in range(36):
        cache.prefill_kv_cache(values, layer_idx, keys)

    torch.cuda.synchronize(device)
    query = torch.randn(1, 32, 1, 128, device=device, dtype=torch.bfloat16)
    summaries_before = int(cache.metadata_valid.sum().item())
    checked_layers = 0
    checked_tokens = 0
    for layer_idx in (0, 17, 35):
        block_ids = cache._select_block_ids(layer_idx, query)
        offsets = torch.arange(8, device=device)
        positions = (block_ids.unsqueeze(-1) * 8 + offsets).reshape(1, 8, -1)
        gathered_keys, gathered_values = cache.get_key_value_cache(layer_idx, positions)
        expected_keys = keys.gather(
            2, positions.unsqueeze(-1).expand(-1, -1, -1, 128)
        )
        expected_values = values.gather(
            2, positions.unsqueeze(-1).expand(-1, -1, -1, 128)
        )
        torch.testing.assert_close(gathered_keys, expected_keys)
        torch.testing.assert_close(gathered_values, expected_values)
        checked_layers += 1
        checked_tokens += int(positions.shape[-1])

    torch.cuda.synchronize(device)
    gaps = cache.query_robust_duality_gap_stats()
    env = {
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(capability),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "asset_shape": [36, 8, 32, 128],
        "asset_sha256": "189b839536e53dac532b032504311b803438d0a968a5aed1db90223f0e76fd68",
        "solver_iters": 24,
    }
    summary = {
        "status": "passed",
        "metadata_valid_page_heads": summaries_before,
        "checked_layers": checked_layers,
        "checked_tokens_per_layer": checked_tokens // max(1, checked_layers),
        "duality_gap": gaps,
    }
    print(f"QR_ENV_JSON={json.dumps(env, sort_keys=True)}", flush=True)
    print(f"QR_SUMMARY_JSON={json.dumps(summary, sort_keys=True)}", flush=True)
    return {"environment": env, "summary": summary}


@APP.local_entrypoint()
def main() -> None:
    result = qr_smoke.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
