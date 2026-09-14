"""Fused CUDA/Triton decode scorer for Query-Robust page routing."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by reference-only installs
    triton = None
    tl = None


if triton is not None:

    @triton.jit(
        do_not_specialize=["first_page", "n_pages", "out_stride_b"]
    )
    def _query_robust_page_score_kernel(
        query,
        landmark,
        bias,
        epsilon,
        metadata_valid,
        output,
        first_page,
        n_pages,
        q_stride_b,
        q_stride_h,
        q_stride_d,
        l_stride_p,
        l_stride_h,
        l_stride_d,
        s_stride_p,
        s_stride_h,
        e_stride_p,
        e_stride_h,
        v_stride_p,
        v_stride_h,
        out_stride_b,
        scale,
        alpha,
        heads: tl.constexpr,
        groups: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles = tl.cdiv(n_pages, BLOCK_P)
        batch = pid // tiles
        tile = pid - batch * tiles

        relative_page = tile * BLOCK_P + tl.arange(0, BLOCK_P)
        page = first_page + relative_page
        page_mask = relative_page < n_pages
        d = tl.arange(0, BLOCK_D)
        d_mask = d < dim

        page_scores = tl.full((BLOCK_P,), -float("inf"), tl.float32)
        page_valid = tl.full((BLOCK_P,), True, tl.int1)

        for kv_head in range(heads):
            valid_ptr = metadata_valid + page * v_stride_p + kv_head * v_stride_h
            head_valid = tl.load(valid_ptr, mask=page_mask, other=0)
            page_valid = page_valid & head_valid

            head_scores = tl.full((BLOCK_P,), -float("inf"), tl.float32)
            landmark_ptr = (
                landmark
                + page[:, None] * l_stride_p
                + kv_head * l_stride_h
                + d[None, :] * l_stride_d
            )
            page_landmark = tl.load(
                landmark_ptr,
                mask=page_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            for group in range(groups):
                query_head = kv_head * groups + group
                query_ptr = (
                    query
                    + batch * q_stride_b
                    + query_head * q_stride_h
                    + d * q_stride_d
                )
                q = tl.load(query_ptr, mask=d_mask, other=0.0).to(tl.float32)
                dot = tl.sum(page_landmark * q[None, :], axis=1)
                head_scores = tl.maximum(head_scores, dot)

            bias_ptr = bias + page * s_stride_p + kv_head * s_stride_h
            epsilon_ptr = epsilon + page * e_stride_p + kv_head * e_stride_h
            head_bias = tl.load(bias_ptr, mask=page_mask, other=0.0).to(tl.float32)
            head_epsilon = tl.load(
                epsilon_ptr, mask=page_mask, other=0.0
            ).to(tl.float32)
            head_scores = head_scores * scale + head_bias + alpha * head_epsilon
            page_scores = tl.maximum(page_scores, head_scores)

        result = tl.where(page_valid, page_scores, float("inf"))
        tl.store(
            output + batch * out_stride_b + relative_page,
            result,
            mask=page_mask,
        )


@torch.inference_mode()
def query_robust_page_scores(
    query: torch.Tensor,
    landmark: torch.Tensor,
    bias: torch.Tensor,
    epsilon: torch.Tensor,
    metadata_valid: torch.Tensor,
    *,
    first_page: int,
    last_page: int,
    scale: float,
    alpha: float,
) -> torch.Tensor:
    """Score a contiguous QR page range with one shared score per page.

    Args:
        query: Query states with shape ``[B, QH, D]``.
        landmark: Stored BF16 QR landmarks with shape ``[P, KVH, D]``.
        bias: FP32 entropy bias with shape ``[P, KVH]``.
        epsilon: FP32 certificate radius with shape ``[P, KVH]``.
        metadata_valid: Boolean metadata mask with shape ``[P, KVH]``.
        first_page: Inclusive page offset into the metadata tensors.
        last_page: Exclusive page offset into the metadata tensors.
        scale: Query/key scale used by the model.
        alpha: QR certificate multiplier.

    Returns:
        FP32 scores with shape ``[B, last_page - first_page]``. Invalid pages
        are assigned positive infinity, matching the reference scorer.
    """
    if triton is None:
        raise RuntimeError("Query-Robust Triton scorer requires Triton")
    tensors = (query, landmark, bias, epsilon, metadata_valid)
    if any(not tensor.is_cuda for tensor in tensors):
        raise RuntimeError("Query-Robust Triton scorer requires CUDA tensors")
    if query.ndim != 3 or landmark.ndim != 3:
        raise ValueError("QR scorer expects query [B,QH,D] and landmark [P,H,D]")
    if bias.ndim != 2 or epsilon.shape != bias.shape or metadata_valid.shape != bias.shape:
        raise ValueError("QR scalar metadata must all have shape [P,H]")
    if not all(tensor.is_floating_point() for tensor in (query, landmark, bias, epsilon)):
        raise ValueError("QR scorer query and metadata must be floating point")
    if metadata_valid.dtype != torch.bool:
        raise ValueError("QR metadata_valid must be bool")

    batch, query_heads, dim = map(int, query.shape)
    pages, kv_heads, landmark_dim = map(int, landmark.shape)
    if tuple(bias.shape) != (pages, kv_heads):
        raise ValueError("QR scalar metadata must match landmark pages and heads")
    if landmark_dim != dim or query_heads % kv_heads:
        raise ValueError("QR query heads must be divisible by local KV heads")
    if not 0 <= int(first_page) < int(last_page) <= pages:
        raise ValueError("QR page range must be non-empty and within metadata pages")
    if not torch.isfinite(torch.tensor(float(scale))) or not torch.isfinite(
        torch.tensor(float(alpha))
    ):
        raise ValueError("QR scale and alpha must be finite")

    n_pages = int(last_page - first_page)
    block_p = 64 if n_pages < 8192 else 128
    output = torch.empty((batch, n_pages), device=query.device, dtype=torch.float32)
    grid = (batch * triton.cdiv(n_pages, block_p),)
    _query_robust_page_score_kernel[grid](
        query,
        landmark,
        bias,
        epsilon,
        metadata_valid,
        output,
        int(first_page),
        n_pages,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        landmark.stride(0),
        landmark.stride(1),
        landmark.stride(2),
        bias.stride(0),
        bias.stride(1),
        epsilon.stride(0),
        epsilon.stride(1),
        metadata_valid.stride(0),
        metadata_valid.stride(1),
        output.stride(0),
        float(scale),
        float(alpha),
        heads=kv_heads,
        groups=query_heads // kv_heads,
        dim=dim,
        BLOCK_P=block_p,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=4,
    )
    return output


__all__ = ["query_robust_page_scores"]
