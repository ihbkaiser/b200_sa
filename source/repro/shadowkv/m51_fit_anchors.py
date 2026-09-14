#!/usr/bin/env python
"""
Fit the offline part of M51 (Routed Multi-Anchor PQ LSE Retrieval) once, so the
per-context work at inference time is only what genuinely depends on the
context.

Two things come out of the calibration query bank and are context-independent,
because both live in query space:
  * the anchors u_m, per (layer, kv_head), by farthest-first init + k-means --
    the same routine the probe uses (probe_routed_tangent_m51.fit_query_anchors);
  * the calibration queries themselves, kept so the per-context envelope can be
    calibrated against this context's blocks without capturing anything new.

The bank is split exactly as the probe splits it: the first half of the prompts
tune the anchors, the second half calibrates the envelope. Keeping that split
is the whole point -- anchors fitted on the queries that later measure coverage
would make the coverage number meaningless.

  $PY repro/shadowkv/m51_fit_anchors.py \
      --bank /storage/baonn/certified_sparse_diverse_qwen3_4b_32k_20260828 \
      --anchors 8 --out /storage/baonn/shadowkv_quest_20260827/_m51/anchors_qwen3_m8.npz
"""

import argparse
import glob
import math
import os

import numpy as np
import torch


def fit_query_anchors(points: torch.Tensor, count: int, iterations: int = 20) -> torch.Tensor:
    """Farthest-first initialization then Lloyd iterations.

    Copied in behaviour from repro/certified_sparse/probe_routed_tangent_m51.py
    so the deployed anchors are the ones the probe measured, not a lookalike.
    """
    if count < 1 or count > len(points):
        raise ValueError(f"invalid anchor count {count} for {len(points)} points")
    mean = points.mean(dim=0, keepdim=True)
    first = int(torch.cdist(points, mean).argmin())
    chosen = [first]
    nearest = torch.linalg.vector_norm(points - points[first], dim=-1)
    for _ in range(1, count):
        index = int(nearest.argmax())
        chosen.append(index)
        nearest = torch.minimum(nearest, torch.linalg.vector_norm(points - points[index], dim=-1))
    anchors = points[torch.tensor(chosen, device=points.device)].clone()
    for _ in range(iterations):
        assignment = torch.cdist(points, anchors).argmin(dim=-1)
        updated = anchors.clone()
        for index in range(count):
            members = points[assignment == index]
            if len(members):
                updated[index] = members.mean(dim=0)
        if torch.allclose(updated, anchors, rtol=0, atol=1.0e-12):
            break
        anchors = updated
    return anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="dir of calib_prompt_*.npz")
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.bank, "calib_prompt_*.npz")))
    if len(paths) < 4:
        raise SystemExit(f"expected >= 4 calibration prompts under {args.bank}, found {len(paths)}")

    banks = []
    for p in paths:
        with np.load(p) as z:
            banks.append(z["queries"])          # [layers, steps, kv_heads, groups, dim]
    stacked = np.stack(banks)                    # [prompts, L, S, H, G, D]
    n_prompts, n_layers, n_steps, n_heads, n_groups, dim = stacked.shape
    print(f"bank: {n_prompts} prompts x {n_layers} layers x {n_steps} steps x "
          f"{n_heads} kv heads x {n_groups} groups x {dim} dim")

    mid = n_prompts // 2
    scale = 1.0 / math.sqrt(dim)                 # probe pre-scales queries by 1/sqrt(d)

    anchors = np.zeros((n_layers, n_heads, args.anchors, dim), dtype=np.float32)
    # calibration queries are kept per (layer, head, group): the envelope is
    # calibrated per GQA query group, so the group index must survive.
    calib = np.zeros((n_layers, n_heads, n_groups, (n_prompts - mid) * n_steps, dim),
                     dtype=np.float16)

    for layer in range(n_layers):
        for head in range(n_heads):
            tune = torch.from_numpy(
                stacked[:mid, layer, :, head].reshape(-1, dim).astype(np.float32)
            ).to(args.device) * scale
            anchors[layer, head] = fit_query_anchors(tune, args.anchors).cpu().numpy()
            for g in range(n_groups):
                calib[layer, head, g] = (
                    stacked[mid:, layer, :, head, g].reshape(-1, dim).astype(np.float32) * scale
                ).astype(np.float16)
        if layer % 6 == 0:
            print(f"  layer {layer}/{n_layers} done")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, anchors=anchors, calib=calib,
             n_anchors=args.anchors, dim=dim, n_groups=n_groups,
             tune_prompts=mid, calib_prompts=n_prompts - mid,
             bank=os.path.basename(args.bank.rstrip("/")))
    print(f"\nwrote {args.out}")
    print(f"  anchors {anchors.shape}  calib {calib.shape} "
          f"({calib.shape[3]} calib queries per (layer, head, group))")


if __name__ == "__main__":
    main()
