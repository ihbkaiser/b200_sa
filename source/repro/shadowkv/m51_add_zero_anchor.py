#!/usr/bin/env python
"""Append the zero anchor to a bank.

The tangent at u = 0 is the PLAIN MEAN of a block's keys, with offset
log(leaf_size) -- which is exactly ShadowKV's landmark score. Since every anchor
yields a valid bound and the scan takes the minimum over branches, adding this
one branch makes M51's block score a superset of ShadowKV's: it can only be
tighter, never looser, and it costs one more anchor slot in the scan.

  $PY repro/shadowkv/m51_add_zero_anchor.py in.npz out.npz
"""
import sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
z = dict(np.load(src))
a = z["anchors"]                                   # [L, H, M, D]
z["anchors"] = np.concatenate([a, np.zeros_like(a[:, :, :1])], axis=2)
z["n_anchors"] = np.array(z["anchors"].shape[2])
np.savez(dst, **z)
print(f"{a.shape[2]} -> {z['anchors'].shape[2]} anchors, last one is zero  ->  {dst}")
