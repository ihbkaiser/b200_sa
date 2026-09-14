#!/usr/bin/env python3
"""Validate the vendored Qwen3 Query-Robust vertex artifact offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from models.query_robust import load_query_robust_asset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        type=Path,
        default=Path(__file__).parents[1]
        / "artifacts/query_robust/qwen3_4b_128k/qwen3_4b_qr_vertices_m32_128k.pt",
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or args.asset.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.asset.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise SystemExit(
            f"SHA-256 mismatch: expected={manifest['sha256']} actual={digest}"
        )
    shape = manifest["shape"]
    asset = load_query_robust_asset(
        args.asset,
        num_layers=shape[0],
        global_num_kv_heads=shape[1],
        head_dim=shape[3],
        num_vertices=shape[2],
        tensor_parallel_rank=0,
        tensor_parallel_size=manifest["tp_world_size"],
        expected_model_id=manifest["model_id"],
        expected_model_fingerprint=manifest["model_revision"],
        expected_sha256=manifest["sha256"],
    )
    result = {
        "status": "passed",
        "asset": str(args.asset),
        "sha256": digest,
        "shape": list(asset.vertices.shape),
        "dtype": str(asset.vertices.dtype),
        "valid_vertices": int(asset.num_valid_vertices.min()),
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
