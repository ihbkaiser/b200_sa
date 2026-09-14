"""Fused decode router for Quest min/max page summaries."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - guarded by the public wrapper
    triton = None
    tl = None


if triton is not None:

    @triton.jit(
        do_not_specialize=[
            "first_page", "n_pages", "o_stride_b", "o_stride_h"
        ]
    )
    def _quest_page_score_kernel(
        query,
        page_min,
        page_max,
        output,
        first_page,
        n_pages,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_g: tl.constexpr,
        q_stride_d: tl.constexpr,
        p_stride_b: tl.constexpr,
        p_stride_h: tl.constexpr,
        p_stride_n: tl.constexpr,
        p_stride_d: tl.constexpr,
        o_stride_b,
        o_stride_h,
        o_stride_n: tl.constexpr,
        heads: tl.constexpr,
        groups: tl.constexpr,
        dim: tl.constexpr,
        REDUCE_MAX: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_head = tl.program_id(1)
        batch = batch_head // heads
        head = batch_head - batch * heads
        relative = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        page = first_page + relative
        page_mask = relative < n_pages
        d = tl.arange(0, BLOCK_D)
        d_mask = d < dim

        base = (
            batch * p_stride_b
            + head * p_stride_h
            + page[:, None] * p_stride_n
            + d[None, :] * p_stride_d
        )
        minimum = tl.load(
            page_min + base,
            mask=page_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        maximum = tl.load(
            page_max + base,
            mask=page_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        reduced = tl.full((BLOCK_N,), -float("inf"), tl.float32)
        if not REDUCE_MAX:
            reduced = tl.zeros((BLOCK_N,), tl.float32)
        for group in range(groups):
            q_ptr = (
                query
                + batch * q_stride_b
                + head * q_stride_h
                + group * q_stride_g
                + d * q_stride_d
            )
            q = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)
            contribution = tl.maximum(
                minimum * q[None, :], maximum * q[None, :]
            )
            score = tl.sum(contribution, axis=1)
            if REDUCE_MAX:
                reduced = tl.maximum(reduced, score)
            else:
                reduced += score

        out = (
            output
            + batch * o_stride_b
            + head * o_stride_h
            + relative * o_stride_n
        )
        tl.store(out, reduced, mask=page_mask)


@torch.inference_mode()
def quest_page_scores(
    query: torch.Tensor,
    page_min: torch.Tensor,
    page_max: torch.Tensor,
    *,
    first_page: int,
    last_page: int,
    group_reduce: str,
) -> torch.Tensor:
    """Return fused Quest scores with shape ``[B,H,N]``.

    ``query`` is ``[B,H,G,1,D]`` while both summary tensors are
    ``[B,H,max_pages,D]``.  Reduction over dimensions and GQA query groups is
    performed inside one kernel, avoiding the very large broadcast tensor
    materialized by the reference PyTorch expression.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not query.is_cuda:
        raise ValueError("the fused Quest router requires CUDA tensors")
    if query.shape[-2] != 1:
        raise ValueError("the fused Quest router supports decode q_len=1")
    if group_reduce not in {"max", "sum"}:
        raise ValueError("group_reduce must be max or sum")
    if page_min.shape != page_max.shape:
        raise ValueError("Quest min/max summary shapes differ")

    batch, heads, groups, _, dim = query.shape
    n_pages = int(last_page - first_page)
    if n_pages <= 0:
        raise ValueError("empty Quest page range")
    output = torch.empty(
        (batch, heads, n_pages), device=query.device, dtype=torch.float32
    )
    block_n = 32 if n_pages < 8192 else 64
    grid = (triton.cdiv(n_pages, block_n), batch * heads)
    _quest_page_score_kernel[grid](
        query,
        page_min,
        page_max,
        output,
        first_page,
        n_pages,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(4),
        page_min.stride(0),
        page_min.stride(1),
        page_min.stride(2),
        page_min.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        heads,
        groups,
        dim,
        REDUCE_MAX=group_reduce == "max",
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=4,
    )
    return output
