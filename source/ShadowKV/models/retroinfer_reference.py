"""Small, auditable reference for RetroInfer's published attention logic.

This is an independent PyTorch implementation of the algorithmic contract in
the RetroInfer paper and its MIT-licensed reference implementation.  It is for
correctness/equivalence tests, not a replacement for the authors' wave-buffer
and fused CUDA kernels.  Do not register it as a production ``retroinfer``
method until its cluster choices and outputs have been compared on captured
model tensors.

RetroInfer copyright: Microsoft Corporation; official code is MIT licensed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RetroInferIndex:
    """Reference cluster index for one set of grouped-query KV heads."""

    centroids: torch.Tensor     # [groups, clusters, dim], arithmetic means
    value_sum: torch.Tensor     # [groups, clusters, value_dim]
    assignments: torch.Tensor   # [groups, tokens]
    cluster_size: torch.Tensor  # [groups, clusters]


@torch.inference_mode()
def segmented_spherical_kmeans(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    num_centroids: int,
    num_segments: int = 1,
    num_iters: int = 10,
) -> RetroInferIndex:
    """Reference form of RetroInfer's segmented spherical k-means.

    Training iterations assign by inner product and normalize updated
    centroids.  As in the official implementation, the final assignment is
    global across the segment-trained centroids and the returned centroids are
    unnormalized arithmetic means.  Empty centroids retain their prior value.
    """
    if keys.ndim != 3 or values.ndim != 3:
        raise ValueError("keys and values must have shape [groups,tokens,dim]")
    if keys.shape[:2] != values.shape[:2]:
        raise ValueError("keys and values must have matching groups and tokens")
    groups, token_count, dim = keys.shape
    if not 0 < num_centroids <= token_count:
        raise ValueError("num_centroids must lie in [1, token_count]")
    if num_centroids % num_segments or token_count % num_segments:
        raise ValueError("tokens and centroids must be divisible by num_segments")
    if num_iters < 1:
        raise ValueError("num_iters must be positive")

    if keys.is_cuda:
        from .retroinfer_triton import segmented_spherical_kmeans_triton

        centroids, value_sum, assignments, cluster_size = (
            segmented_spherical_kmeans_triton(
                keys,
                values,
                num_centroids=num_centroids,
                num_segments=num_segments,
                num_iters=num_iters,
            )
        )
        return RetroInferIndex(
            centroids=centroids.float(),
            value_sum=value_sum.float(),
            assignments=assignments,
            cluster_size=cluster_size,
        )

    # This matches the uniform midpoint initialization of the official code.
    step = token_count / num_centroids
    initial_ids = (
        torch.arange(num_centroids, device=keys.device, dtype=torch.float32)
        * step
        + step / 2
    ).long()
    centroids = keys.float().index_select(1, initial_ids)
    tokens_per_segment = token_count // num_segments
    centroids_per_segment = num_centroids // num_segments
    segment_keys = keys.float().reshape(
        groups, num_segments, tokens_per_segment, dim
    ).reshape(groups * num_segments, tokens_per_segment, dim)
    segment_centroids = centroids.reshape(
        groups * num_segments, centroids_per_segment, dim
    )

    for _ in range(num_iters - 1):
        assignment = torch.einsum(
            "gnd,gkd->gnk", segment_keys, segment_centroids
        ).argmax(-1)
        counts = torch.zeros(
            assignment.shape[0], centroids_per_segment,
            device=keys.device, dtype=torch.float32,
        )
        counts.scatter_add_(1, assignment, torch.ones_like(assignment).float())
        sums = torch.zeros_like(segment_centroids)
        sums.scatter_add_(
            1,
            assignment[..., None].expand(-1, -1, dim),
            segment_keys,
        )
        means = sums / counts.clamp_min(1)[..., None]
        updated = torch.where(
            (counts > 0)[..., None], means, segment_centroids
        )
        segment_centroids = nn.functional.normalize(updated, dim=-1, eps=1e-12)

    # The official final pass reshapes segments back and assigns against every
    # trained centroid, then returns the (unnormalized) cluster means.
    trained = segment_centroids.reshape(groups, num_centroids, dim)
    assignments = torch.einsum("gnd,gkd->gnk", keys.float(), trained).argmax(-1)
    cluster_size = torch.zeros(
        groups, num_centroids, device=keys.device, dtype=torch.long
    )
    cluster_size.scatter_add_(
        1, assignments, torch.ones_like(assignments, dtype=torch.long)
    )
    key_sum = torch.zeros_like(trained)
    key_sum.scatter_add_(
        1, assignments[..., None].expand(-1, -1, dim), keys.float()
    )
    centroids = torch.where(
        (cluster_size > 0)[..., None],
        key_sum / cluster_size.clamp_min(1)[..., None],
        trained,
    )
    value_sum = torch.zeros(
        groups, num_centroids, values.shape[-1],
        device=values.device, dtype=torch.float32,
    )
    value_sum.scatter_add_(
        1,
        assignments[..., None].expand(-1, -1, values.shape[-1]),
        values.float(),
    )

    return RetroInferIndex(
        centroids=centroids,
        value_sum=value_sum,
        assignments=assignments,
        cluster_size=cluster_size,
    )


def _validate_cluster_ids(
    ids: torch.Tensor, groups: int, cluster_count: int, name: str
) -> torch.Tensor:
    if ids.ndim == 1:
        ids = ids.unsqueeze(0).expand(groups, -1)
    if ids.ndim != 2 or ids.shape[0] != groups:
        raise ValueError(f"{name} must have shape [groups,count] or [count]")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= cluster_count):
        raise ValueError(f"{name} contains an invalid cluster id")
    return ids.long()


@torch.inference_mode()
def rank_retroinfer_clusters(
    queries: torch.Tensor,
    index: RetroInferIndex,
) -> torch.Tensor:
    """Rank centroids as RetroInfer does for grouped-query attention.

    Per-query softmax probabilities over centroids are summed over all query
    heads sharing a KV head.  Empty clusters are masked.
    """
    if queries.ndim == 2:
        queries = queries[:, None, :]
    if queries.ndim != 3 or queries.shape[0] != index.centroids.shape[0]:
        raise ValueError("queries must have shape [groups,q_heads,dim]")
    logits = torch.einsum(
        "gqd,gkd->gqk", queries.float(), index.centroids.float()
    ) / math.sqrt(queries.shape[-1])
    probabilities = torch.softmax(logits, dim=-1).sum(dim=1)
    probabilities.masked_fill_(index.cluster_size == 0, float("-inf"))
    return torch.argsort(probabilities, dim=-1, descending=True, stable=True)


@torch.inference_mode()
def retroinfer_partition_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    index: RetroInferIndex,
    *,
    retrieve_ids: torch.Tensor,
    estimation_ids: torch.Tensor | None = None,
    steady_keys: torch.Tensor | None = None,
    steady_values: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Merge RetroInfer's exact and estimated zones in one stable softmax.

    Tokens in ``retrieve_ids`` plus the optional steady zone are evaluated
    exactly.  Each estimation cluster contributes

    ``exp(q @ centroid) * value_sum`` to the numerator and
    ``cluster_size * exp(q @ centroid)`` to the denominator.

    Clusters absent from both sets make no contribution.  The two sets must be
    disjoint, which prevents accidental double counting.
    """
    if queries.ndim == 2:
        queries = queries[:, None, :]
    if queries.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
        raise ValueError("queries/keys/values require [groups,items,dim] shapes")
    groups, q_heads, dim = queries.shape
    if keys.shape[:2] != values.shape[:2] or keys.shape[0] != groups:
        raise ValueError("keys and values do not match query groups")
    if index.assignments.shape != keys.shape[:2]:
        raise ValueError("index assignments do not match keys")
    if steady_keys is None and steady_values is not None:
        raise ValueError("steady_keys and steady_values must be supplied together")
    if steady_keys is not None:
        if steady_values is None or steady_keys.shape[:2] != steady_values.shape[:2]:
            raise ValueError("steady key/value shapes differ")
        if steady_keys.shape[0] != groups:
            raise ValueError("steady zone has the wrong group count")
    scale = dim**-0.5 if scale is None else scale
    cluster_count = index.centroids.shape[1]
    retrieve_ids = _validate_cluster_ids(
        retrieve_ids, groups, cluster_count, "retrieve_ids"
    )
    if estimation_ids is None:
        estimation_ids = torch.empty(groups, 0, device=keys.device, dtype=torch.long)
    estimation_ids = _validate_cluster_ids(
        estimation_ids, groups, cluster_count, "estimation_ids"
    )
    if retrieve_ids.shape[1] and estimation_ids.shape[1]:
        overlap = (
            retrieve_ids[:, :, None] == estimation_ids[:, None, :]
        ).any()
        if bool(overlap):
            raise ValueError("retrieval and estimation zones overlap")

    # We collect logits and already-weighted values, then perform one stable
    # normalization.  For estimation clusters, log(cluster_size) belongs only
    # in the denominator; dividing value_sum by size makes the numerator exact.
    outputs = []
    for group in range(groups):
        zone_logits = []
        zone_values = []
        if steady_keys is not None and steady_keys.shape[1]:
            zone_logits.append(
                torch.einsum(
                    "qd,nd->qn", queries[group].float(), steady_keys[group].float()
                ) * scale
            )
            zone_values.append(steady_values[group].float())

        # Keep this operation on-device.  The earlier audit implementation
        # converted every cluster id to a Python scalar, which introduced
        # hundreds of CUDA synchronizations per layer without changing the
        # mathematical partition.
        exact_mask = (
            index.assignments[group, :, None] == retrieve_ids[group, None, :]
        ).any(dim=-1)
        if torch.any(exact_mask):
            zone_logits.append(
                torch.einsum(
                    "qd,nd->qn", queries[group].float(), keys[group, exact_mask].float()
                ) * scale
            )
            zone_values.append(values[group, exact_mask].float())

        if estimation_ids.shape[1]:
            estimate_ids = estimation_ids[group]
            sizes = index.cluster_size[group].index_select(0, estimate_ids)
            nonempty = sizes > 0
            estimate_ids = estimate_ids[nonempty]
            sizes = sizes[nonempty].float()
            if estimate_ids.numel():
                estimate_centroids = index.centroids[group].index_select(
                    0, estimate_ids
                ).float()
                estimate_values = index.value_sum[group].index_select(
                    0, estimate_ids
                ).float() / sizes[:, None]
                zone_logits.append(
                    torch.einsum(
                        "qd,nd->qn", queries[group].float(), estimate_centroids
                    ) * scale
                    + sizes.log()[None]
                )
                zone_values.append(estimate_values)

        if not zone_logits:
            raise ValueError("attention partition contains no non-empty zone")
        logits = torch.cat(zone_logits, dim=-1)
        merged_values = torch.cat(zone_values, dim=0)
        outputs.append(torch.softmax(logits, dim=-1) @ merged_values)
    return torch.stack(outputs)


@torch.inference_mode()
def retroinfer_reference_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    index: RetroInferIndex,
    *,
    retrieve_clusters: int,
    estimation_clusters: int,
    steady_keys: torch.Tensor | None = None,
    steady_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rank clusters, split retrieval/estimation zones, and run attention."""
    if retrieve_clusters < 0 or estimation_clusters < 0:
        raise ValueError("zone sizes cannot be negative")
    if retrieve_clusters + estimation_clusters > index.centroids.shape[1]:
        raise ValueError("requested more clusters than the index contains")
    ranking = rank_retroinfer_clusters(queries, index)
    retrieve = ranking[:, :retrieve_clusters]
    estimate = ranking[
        :, retrieve_clusters : retrieve_clusters + estimation_clusters
    ]
    return retroinfer_partition_attention(
        queries,
        keys,
        values,
        index,
        retrieve_ids=retrieve,
        estimation_ids=estimate,
        steady_keys=steady_keys,
        steady_values=steady_values,
    )
