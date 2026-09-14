"""Triton clustering primitives adapted from Microsoft RetroInfer.

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Adapted from ``cache_hub/kmeans.py`` at RetroInfer author commit
75829e630122d4ea6f568dcd001405698bc2db84. The reverse-index materialization
is omitted because the ShadowKV accuracy harness consumes token assignments
directly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _assign_kernel(
    data, centroids, data_sum, data_count, assignments,
    stride_dz, stride_dn, stride_dd,
    stride_cz, stride_ck, stride_cd,
    stride_sz, stride_sk, stride_sd,
    stride_nz, stride_nk,
    stride_az, stride_an,
    num_tokens, num_centroids,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    batch = tl.program_id(1)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < num_tokens
    data_ptr = data + batch * stride_dz + offs_n[:, None] * stride_dn + offs_d[None] * stride_dd
    centroid_ptr = centroids + batch * stride_cz + offs_k[None] * stride_ck + offs_d[:, None] * stride_cd
    sum_ptr = data_sum + batch * stride_sz + offs_d[None] * stride_sd
    count_ptr = data_count + batch * stride_nz
    assignment_ptr = assignments + batch * stride_az + offs_n * stride_an
    vectors = tl.load(data_ptr, mask=mask_n[:, None], other=0.0)
    best_value = tl.zeros([BLOCK_N], tl.float32) - float("inf")
    best_index = tl.zeros([BLOCK_N], tl.int32)
    for start_k in tl.range(0, num_centroids, BLOCK_K):
        mask_k = start_k + offs_k < num_centroids
        centers = tl.load(centroid_ptr, mask=mask_k[None], other=0.0)
        product = tl.dot(vectors, centers).to(tl.float32)
        product = tl.where(mask_k[None], product, float("-inf"))
        value, index = tl.max(product, axis=1, return_indices=True)
        index += start_k
        best_index = tl.where(value > best_value, index, best_index)
        best_value = tl.maximum(value, best_value)
        centroid_ptr += BLOCK_K * stride_ck
    tl.store(assignment_ptr, best_index, mask=mask_n)
    tl.atomic_add(
        sum_ptr + best_index[:, None] * stride_sk,
        vectors.to(tl.float32), mask=mask_n[:, None], sem="relaxed",
    )
    tl.atomic_add(
        count_ptr + best_index * stride_nk,
        tl.zeros_like(best_index) + 1, mask=mask_n, sem="relaxed",
    )


@triton.jit
def _update_kernel(
    centroids, data_sum, data_count,
    stride_cz, stride_ck, stride_cd,
    stride_sz, stride_sk, stride_sd,
    stride_nz, stride_nk,
    num_centroids,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    start_k = tl.program_id(0) * BLOCK_K
    batch = tl.program_id(1)
    offs_k = start_k + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    mask_k = offs_k < num_centroids
    centroid_ptr = centroids + batch * stride_cz + offs_k[:, None] * stride_ck + offs_d[None] * stride_cd
    sum_ptr = data_sum + batch * stride_sz + offs_k[:, None] * stride_sk + offs_d[None] * stride_sd
    count_ptr = data_count + batch * stride_nz + offs_k[:, None] * stride_nk
    sums = tl.load(sum_ptr, mask=mask_k[:, None], other=0.0)
    counts = tl.load(count_ptr, mask=mask_k[:, None], other=0)
    nonempty = counts > 0
    updated = sums / counts
    if NORMALIZE:
        updated /= tl.sqrt(tl.sum(updated * updated, axis=-1, keep_dims=True))
    tl.store(centroid_ptr, updated.to(centroids.type.element_ty), mask=nonempty)


def _train(
    data: torch.Tensor,
    centroids: torch.Tensor,
    *,
    assignments: torch.Tensor | None = None,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, num_tokens, dim = data.shape
    num_centroids = centroids.shape[1]
    data_sum = torch.zeros_like(centroids, dtype=torch.float32)
    data_count = torch.zeros(
        batch, num_centroids, dtype=torch.int32, device=data.device
    )
    if assignments is None:
        assignments = torch.empty(
            batch, num_tokens, dtype=torch.int32, device=data.device
        )
    block_n, block_k = 128, 64
    _assign_kernel[(triton.cdiv(num_tokens, block_n), batch)](
        data, centroids, data_sum, data_count, assignments,
        data.stride(0), data.stride(1), data.stride(2),
        centroids.stride(0), centroids.stride(1), centroids.stride(2),
        data_sum.stride(0), data_sum.stride(1), data_sum.stride(2),
        data_count.stride(0), data_count.stride(1),
        assignments.stride(0), assignments.stride(1),
        num_tokens, num_centroids,
        BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_D=dim,
        num_warps=4, num_stages=2,
    )
    update_k = 128
    _update_kernel[(triton.cdiv(num_centroids, update_k), batch)](
        centroids, data_sum, data_count,
        centroids.stride(0), centroids.stride(1), centroids.stride(2),
        data_sum.stride(0), data_sum.stride(1), data_sum.stride(2),
        data_count.stride(0), data_count.stride(1),
        num_centroids,
        BLOCK_K=update_k, BLOCK_D=dim, NORMALIZE=normalize,
        num_warps=4, num_stages=1,
    )
    return centroids, assignments, data_count


@triton.jit
def _value_sum_kernel(
    values, value_sum, assignments,
    stride_vz, stride_vn, stride_vd,
    stride_sz, stride_sk, stride_sd,
    stride_az, stride_an,
    num_tokens,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    batch = tl.program_id(1)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < num_tokens
    value_ptr = values + batch * stride_vz + offs_n[:, None] * stride_vn + offs_d[None] * stride_vd
    sum_ptr = value_sum + batch * stride_sz + offs_d[None] * stride_sd
    assignment_ptr = assignments + batch * stride_az + offs_n * stride_an
    value = tl.load(value_ptr, mask=mask_n[:, None], other=0.0)
    cluster = tl.load(assignment_ptr, mask=mask_n, other=0)
    tl.atomic_add(
        sum_ptr + cluster[:, None] * stride_sk,
        value.to(tl.float32), mask=mask_n[:, None], sem="relaxed",
    )


def _sum_values(
    values: torch.Tensor, assignments: torch.Tensor, num_centroids: int
) -> torch.Tensor:
    batch, num_tokens, dim = values.shape
    output = torch.zeros(
        batch, num_centroids, dim, dtype=torch.float32, device=values.device
    )
    block_n = 128
    _value_sum_kernel[(triton.cdiv(num_tokens, block_n), batch)](
        values, output, assignments,
        values.stride(0), values.stride(1), values.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        assignments.stride(0), assignments.stride(1),
        num_tokens, BLOCK_N=block_n, BLOCK_D=dim,
        num_warps=4, num_stages=1,
    )
    return output


@torch.inference_mode()
def segmented_spherical_kmeans_triton(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    num_centroids: int,
    num_segments: int,
    num_iters: int,
):
    """Return centroids, value sums, assignments, and cluster populations."""
    if not keys.is_cuda or not values.is_cuda:
        raise ValueError("Triton clustering requires CUDA tensors")
    groups, num_tokens, dim = keys.shape
    step = num_tokens / num_centroids
    initial = (
        torch.arange(num_centroids, device=keys.device, dtype=torch.float32)
        * step + step / 2
    ).long()
    centroids = keys.index_select(1, initial)
    tokens_per_segment = num_tokens // num_segments
    centroids_per_segment = num_centroids // num_segments
    data = keys[:, : tokens_per_segment * num_segments].reshape(
        groups * num_segments, tokens_per_segment, dim
    ).contiguous()
    centroids = centroids.reshape(
        groups * num_segments, centroids_per_segment, dim
    ).contiguous()
    scratch = torch.empty(
        data.shape[:2], device=keys.device, dtype=torch.int32
    )
    for _ in range(num_iters - 1):
        centroids, _, _ = _train(
            data, centroids, assignments=scratch, normalize=True
        )
    centroids = centroids.reshape(groups, num_centroids, dim).contiguous()
    centroids, assignments, populations = _train(
        keys.contiguous(), centroids, normalize=False
    )
    value_sum = _sum_values(values.contiguous(), assignments, num_centroids)
    return centroids, value_sum, assignments.long(), populations.long()
