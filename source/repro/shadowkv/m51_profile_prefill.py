#!/usr/bin/env python
"""
Where does M51's per-context prefill time actually go?

Three phases build the metadata, and the writeup guessed the envelope
(512 x T x d) was the bottleneck. This times them separately, per layer, so the
guess can be checked instead of repeated:

  tangents   anchor logits, softmax, weighted centroid, offsets   ~ M*T*d
  pq         k-means codebook over the tangents                   iterative
  envelope   calib queries against this context's blocks          ~ Ncalib*T*d

  CUDA_VISIBLE_DEVICES=0 $PY repro/shadowkv/m51_profile_prefill.py --datalens 8192,32768
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ShadowKV"))
from models.routed_pq_cache import _batched_kmeans  # noqa: E402


def timed(fn):
    torch.cuda.synchronize(); t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default=os.environ.get("SHADOWKV_M51_ANCHORS_QWEN3"))
    ap.add_argument("--datalens", default="8192,16384,32768")
    ap.add_argument("--layers", type=int, default=36, help="layers in the model, for the total")
    ap.add_argument("--leaf", type=int, default=8)
    ap.add_argument("--subdim", type=int, default=8)
    ap.add_argument("--codes", type=int, default=64)
    args = ap.parse_args()

    z = np.load(args.anchors)
    anchors_all = torch.from_numpy(z["anchors"]).cuda().float()     # [L, H, M, D]
    calib_all = torch.from_numpy(z["calib"]).cuda().float()         # [L, H, G, N, D]
    _, H, M, D = anchors_all.shape
    G, Nc = calib_all.shape[2], calib_all.shape[3]
    n_sub = D // args.subdim
    print(f"heads {H} anchors {M} dim {D} groups {G} calib/group {Nc} "
          f"-> {G*Nc} calibration queries per (layer, head)\n")

    print(f"{'T':>7} {'tangents':>10} {'pq kmeans':>11} {'envelope':>10} {'layer':>8} "
          f"{'x{} layers'.format(args.layers):>12}")
    print("-" * 64)

    for T in [int(x) for x in args.datalens.split(",")]:
        torch.manual_seed(0)
        L = T // args.leaf
        keys = torch.randn(H, T, D, device="cuda") * (1.0 / math.sqrt(D))
        leaves = keys.view(H, L, args.leaf, D)
        anchors = anchors_all[0]
        calib = calib_all[0]

        def tangents():
            al = torch.einsum('hmd,hltd->hmlt', anchors, leaves)
            w = torch.softmax(al, dim=-1)
            c = torch.einsum('hmlt,hltd->hmld', w, leaves)
            off = torch.logsumexp(al, dim=-1) - torch.einsum('hmd,hmld->hml', anchors, c)
            return c, off

        (centers, offsets), t_tan = timed(tangents)

        def pq():
            flat = centers.reshape(H, M * L, D)
            recon = torch.empty_like(flat)
            for s in range(n_sub):
                sl = slice(s * args.subdim, (s + 1) * args.subdim)
                cb, code = _batched_kmeans(flat[:, :, sl], args.codes)
                recon[:, :, sl] = torch.gather(cb, 1, code.unsqueeze(-1).expand(-1, -1, args.subdim))
            return recon.view(H, M, L, D)

        recon, t_pq = timed(pq)

        def envelope():
            out = torch.zeros(H, M, G, L, device="cuda")
            route = torch.cdist(calib.reshape(H, G * Nc, D), anchors).argmin(-1).view(H, G, Nc)
            for h in range(H):
                lg = calib[h].reshape(G * Nc, D) @ keys[h].T
                tm = torch.logsumexp(lg.view(G * Nc, L, args.leaf), dim=-1).view(G, Nc, L)
                del lg
                for m in range(M):
                    lin = offsets[h, m].unsqueeze(0) + calib[h].reshape(G * Nc, D) @ recon[h, m].T
                    err = (tm.view(G * Nc, L) - lin).view(G, Nc, L)
                    for g in range(G):
                        sel = route[h, g] == m
                        src = err[g][sel] if sel.any() else err[g]
                        out[h, m, g] = torch.quantile(src, 0.999, dim=0)
            return out

        _, t_env = timed(envelope)
        per_layer = t_tan + t_pq + t_env
        print(f"{T:>7} {t_tan:9.3f}s {t_pq:10.3f}s {t_env:9.3f}s {per_layer:7.3f}s "
              f"{per_layer*args.layers:11.1f}s")
        del keys, leaves, centers, recon
        torch.cuda.empty_cache()

    print("\nshare of the per-layer total:")
    print("  read the three columns above -- the writeup expected the envelope to")
    print("  dominate; whichever column actually does is where a kernel should go.")


if __name__ == "__main__":
    main()
