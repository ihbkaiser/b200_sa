#!/usr/bin/env python
"""Tensor-level gate against RetroInfer's MIT-licensed clustering kernel.

The synthetic clusters are deliberately well separated so the check measures
algorithmic agreement rather than unstable label changes at a decision tie.
The official ``weighted_flash_decoding`` package is not vendored by the author
repository; tripartite attention is therefore guarded separately by the dense
limit tests in ``test_retroinfer_reference.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import torch


def load_author_kmeans(root: str):
    path = os.path.join(root, "cache_hub", "kmeans.py")
    spec = importlib.util.spec_from_file_location("retroinfer_author_kmeans", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load author k-means: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.segment_k_means


def reconstruct_assignments(clusters, sizes, token_count):
    groups, cluster_count, _ = clusters.shape
    assignment = torch.full(
        (groups, token_count), -1, device=clusters.device, dtype=torch.long
    )
    for group in range(groups):
        for cluster in range(cluster_count):
            size = int(sizes[group, cluster])
            if size:
                assignment[group, clusters[group, cluster, :size].long()] = cluster
    if torch.any(assignment < 0):
        raise AssertionError("author reverse index omitted one or more tokens")
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--author-root",
        default=os.environ.get("RETROINFER_AUTHOR_ROOT"),
        help="external RetrievalAttention checkout (or set RETROINFER_AUTHOR_ROOT)",
    )
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if not args.author_root:
        parser.error("--author-root or RETROINFER_AUTHOR_ROOT is required")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the author's Triton k-means")

    shadowkv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ShadowKV"
    )
    sys.path.insert(0, shadowkv)
    from models.retroinfer_reference import segmented_spherical_kmeans

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    groups, clusters, per_cluster, dim = 2, 8, 16, 128
    token_count = clusters * per_cluster
    directions = torch.zeros(groups, clusters, dim, device=device)
    for group in range(groups):
        for cluster in range(clusters):
            directions[group, cluster, group * clusters + cluster] = 4.0
    keys = (
        directions[:, :, None]
        + 0.01 * torch.randn(groups, clusters, per_cluster, dim, device=device)
    ).reshape(groups, token_count, dim).to(torch.bfloat16).contiguous()
    values = torch.randn_like(keys)

    ours = segmented_spherical_kmeans(
        keys, values,
        num_centroids=clusters, num_segments=2, num_iters=4,
    )
    author = load_author_kmeans(args.author_root)
    centroids, value_sum, members, sizes = author(
        keys, values,
        num_centroids=clusters, num_iters=4, num_segments=2,
    )
    author_assignment = reconstruct_assignments(members, sizes, token_count)
    if not torch.equal(ours.assignments, author_assignment):
        agreement = float((ours.assignments == author_assignment).float().mean())
        raise AssertionError(f"cluster assignments differ: agreement={agreement:.6f}")
    torch.testing.assert_close(
        ours.cluster_size.to(sizes.dtype), sizes, rtol=0, atol=0
    )
    torch.testing.assert_close(
        ours.centroids, centroids.float(), rtol=2e-3, atol=2e-3
    )
    torch.testing.assert_close(
        ours.value_sum, value_sum.float(), rtol=2e-2, atol=2e-2
    )
    print(
        "PASS RetroInfer author-reference gate: assignments/sizes exact; "
        "centroids/value sums agree within BF16 accumulation tolerance"
    )
    print(
        "PASS tripartite merge gate: see test_retroinfer_reference.py "
        "(full retrieval equals dense; constant-key estimation equals dense)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
