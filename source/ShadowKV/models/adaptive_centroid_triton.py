"""Fused decode scorers for dense and compact adaptive-centroid routers."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by the explicit gate below
    triton = None
    tl = None


if triton is not None:

    # ``n_blocks`` grows during streaming decode.  Without this opt-out Triton
    # specializes every scalar value and recompiles a new kernel each time a
    # block becomes active, even though the runtime mask already handles the
    # changing tail exactly.
    @triton.jit(
        do_not_specialize=[
            "n_blocks", "o_stride_b", "o_stride_h", "o_stride_g"
        ]
    )
    def _two_slot_lse_kernel(
        query,
        centers,
        log_counts,
        alpha,
        output,
        inv_sqrt_dim,
        n_blocks,
        dim: tl.constexpr,
        groups: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_g: tl.constexpr,
        q_stride_d: tl.constexpr,
        c_stride_b: tl.constexpr,
        c_stride_h: tl.constexpr,
        c_stride_n: tl.constexpr,
        c_stride_r: tl.constexpr,
        c_stride_d: tl.constexpr,
        l_stride_b: tl.constexpr,
        l_stride_h: tl.constexpr,
        l_stride_n: tl.constexpr,
        l_stride_r: tl.constexpr,
        a_stride_b: tl.constexpr,
        a_stride_h: tl.constexpr,
        a_stride_n: tl.constexpr,
        a_stride_r: tl.constexpr,
        o_stride_b,
        o_stride_h,
        o_stride_g,
        o_stride_n: tl.constexpr,
        heads: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        block_group = tl.program_id(1)
        batch = block_group // (heads * groups)
        head_group = block_group - batch * heads * groups
        head = head_group // groups
        group = head_group - head * groups
        n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        d = tl.arange(0, BLOCK_D)
        n_mask = n < n_blocks
        d_mask = d < dim

        q_ptr = (
            query
            + batch * q_stride_b
            + head * q_stride_h
            + group * q_stride_g
            + d * q_stride_d
        )
        q = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)
        qnorm = tl.sum(q * q, axis=0) / dim

        base = (
            centers
            + batch * c_stride_b
            + head * c_stride_h
            + n[:, None] * c_stride_n
            + d[None, :] * c_stride_d
        )
        center0 = tl.load(
            base, mask=n_mask[:, None] & d_mask[None, :], other=0.0
        ).to(tl.float32)
        center1 = tl.load(
            base + c_stride_r,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        dot0 = tl.sum(center0 * q[None, :], axis=1) * inv_sqrt_dim
        dot1 = tl.sum(center1 * q[None, :], axis=1) * inv_sqrt_dim

        meta = log_counts + batch * l_stride_b + head * l_stride_h + n * l_stride_n
        log0 = tl.load(meta, mask=n_mask, other=-float("inf")).to(tl.float32)
        log1 = tl.load(
            meta + l_stride_r, mask=n_mask, other=-float("inf")
        ).to(tl.float32)
        alpha_ptr = alpha + batch * a_stride_b + head * a_stride_h + n * a_stride_n
        alpha0 = tl.load(alpha_ptr, mask=n_mask, other=0.0).to(tl.float32)
        alpha1 = tl.load(
            alpha_ptr + a_stride_r, mask=n_mask, other=0.0
        ).to(tl.float32)
        x0 = dot0 + log0 + qnorm * alpha0
        x1 = dot1 + log1 + qnorm * alpha1
        maximum = tl.maximum(x0, x1)
        lse = maximum + tl.log(tl.exp(x0 - maximum) + tl.exp(x1 - maximum))
        out = (
            output
            + batch * o_stride_b
            + head * o_stride_h
            + group * o_stride_g
            + n * o_stride_n
        )
        tl.store(out, lse, mask=n_mask)

    @triton.jit(
        do_not_specialize=[
            "first_block", "n_blocks", "o_stride_b", "o_stride_h",
            "o_stride_g",
        ]
    )
    def _packed_lse_kernel(
        query,
        centers,
        center_scale,
        log_counts,
        alpha,
        component_start,
        component_count,
        output,
        inv_sqrt_dim,
        first_block,
        n_blocks,
        dim: tl.constexpr,
        groups: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_g: tl.constexpr,
        q_stride_d: tl.constexpr,
        c_stride_b: tl.constexpr,
        c_stride_h: tl.constexpr,
        c_stride_m: tl.constexpr,
        c_stride_d: tl.constexpr,
        s_stride_b: tl.constexpr,
        s_stride_h: tl.constexpr,
        s_stride_m: tl.constexpr,
        l_stride_b: tl.constexpr,
        l_stride_h: tl.constexpr,
        l_stride_m: tl.constexpr,
        a_stride_b: tl.constexpr,
        a_stride_h: tl.constexpr,
        a_stride_m: tl.constexpr,
        st_stride_b: tl.constexpr,
        st_stride_h: tl.constexpr,
        st_stride_n: tl.constexpr,
        ct_stride_b: tl.constexpr,
        ct_stride_h: tl.constexpr,
        ct_stride_n: tl.constexpr,
        o_stride_b,
        o_stride_h,
        o_stride_g,
        o_stride_n: tl.constexpr,
        heads: tl.constexpr,
        CENTER_BITS: tl.constexpr,
        MAX_COMPONENTS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Score CSR-like variable-order mixtures without dequantizing them."""
        block_group = tl.program_id(1)
        batch = block_group // (heads * groups)
        head_group = block_group - batch * heads * groups
        head = head_group // groups
        group = head_group - head * groups
        relative = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        block = first_block + relative
        block_mask = relative < n_blocks
        d = tl.arange(0, BLOCK_D)
        d_mask = d < dim

        q_ptr = (
            query + batch * q_stride_b + head * q_stride_h
            + group * q_stride_g + d * q_stride_d
        )
        q = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)
        qnorm = tl.sum(q * q, axis=0) / dim

        start_ptr = (
            component_start + batch * st_stride_b + head * st_stride_h
            + block * st_stride_n
        )
        count_ptr = (
            component_count + batch * ct_stride_b + head * ct_stride_h
            + block * ct_stride_n
        )
        start = tl.load(start_ptr, mask=block_mask, other=0).to(tl.int32)
        count = tl.load(count_ptr, mask=block_mask, other=0).to(tl.int32)

        running_max = tl.full((BLOCK_N,), -float("inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_N,), tl.float32)
        for slot in range(MAX_COMPONENTS):
            valid = block_mask & (slot < count)
            component = start + slot
            if CENTER_BITS == 4:
                byte_d = d // 2
                packed_ptr = (
                    centers + batch * c_stride_b + head * c_stride_h
                    + component[:, None] * c_stride_m
                    + byte_d[None, :] * c_stride_d
                )
                packed = tl.load(
                    packed_ptr, mask=valid[:, None] & d_mask[None, :], other=0
                ).to(tl.int32)
                shift = (d & 1) * 4
                nibble = (packed >> shift[None, :]) & 15
                center = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
            else:
                center_ptr = (
                    centers + batch * c_stride_b + head * c_stride_h
                    + component[:, None] * c_stride_m
                    + d[None, :] * c_stride_d
                )
                center = tl.load(
                    center_ptr, mask=valid[:, None] & d_mask[None, :], other=0.0
                ).to(tl.float32)

            if CENTER_BITS < 16:
                scale_ptr = (
                    center_scale + batch * s_stride_b + head * s_stride_h
                    + component * s_stride_m
                )
                scale = tl.load(scale_ptr, mask=valid, other=0.0).to(tl.float32)
                center *= scale[:, None]

            dot = tl.sum(center * q[None, :], axis=1) * inv_sqrt_dim
            log_ptr = (
                log_counts + batch * l_stride_b + head * l_stride_h
                + component * l_stride_m
            )
            alpha_ptr = (
                alpha + batch * a_stride_b + head * a_stride_h
                + component * a_stride_m
            )
            log_count = tl.load(log_ptr, mask=valid, other=-float("inf")).to(tl.float32)
            correction = tl.load(alpha_ptr, mask=valid, other=0.0).to(tl.float32)
            value = dot + log_count + qnorm * correction

            next_max = tl.maximum(running_max, value)
            old_weight = tl.where(running_sum > 0.0, tl.exp(running_max - next_max), 0.0)
            new_weight = tl.where(valid, tl.exp(value - next_max), 0.0)
            running_sum = running_sum * old_weight + new_weight
            running_max = next_max

        result = running_max + tl.log(running_sum)
        out = (
            output + batch * o_stride_b + head * o_stride_h
            + group * o_stride_g + relative * o_stride_n
        )
        tl.store(out, result, mask=block_mask)


@torch.inference_mode()
def two_slot_block_logits(
    query: torch.Tensor,
    centers: torch.Tensor,
    log_counts: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Compute component log-sum-exp without materializing both slot logits.

    Shapes are query ``[B,H,G,1,D]``, centers ``[B,H,N,2,D]`` and metadata
    ``[B,H,N,2]``.  The output is FP32 ``[B,H,G,N]``.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not query.is_cuda:
        raise ValueError("Triton router requires CUDA tensors")
    if query.shape[-2] != 1 or centers.shape[-2] != 2:
        raise ValueError("Triton router supports decode q_len=1 and two slots")
    batch, heads, groups, _, dim = query.shape
    blocks = centers.shape[-3]
    output = torch.empty(
        (batch, heads, groups, blocks), device=query.device, dtype=torch.float32
    )
    # Larger block tiles amortize query/metadata traffic at long contexts.
    # This split was profiled on SM86 and remains conservative for smaller
    # contexts where launch latency dominates.
    block_n = 32 if blocks < 8192 else 128
    grid = (triton.cdiv(blocks, block_n), batch * heads * groups)
    _two_slot_lse_kernel[grid](
        query,
        centers,
        log_counts,
        alpha,
        output,
        1.0 / math.sqrt(dim),
        blocks,
        dim,
        groups,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(4),
        centers.stride(0),
        centers.stride(1),
        centers.stride(2),
        centers.stride(3),
        centers.stride(4),
        log_counts.stride(0),
        log_counts.stride(1),
        log_counts.stride(2),
        log_counts.stride(3),
        alpha.stride(0),
        alpha.stride(1),
        alpha.stride(2),
        alpha.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        heads,
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=2,
    )
    return output


@torch.inference_mode()
def packed_block_logits(
    query: torch.Tensor,
    centers: torch.Tensor,
    center_scale: torch.Tensor | None,
    log_counts: torch.Tensor,
    alpha: torch.Tensor,
    component_start: torch.Tensor,
    component_count: torch.Tensor,
    *,
    first_block: int,
    last_block: int,
    center_bits: int,
    max_components: int,
) -> torch.Tensor:
    """Fused compact-center scorer returning FP32 ``[B,H,G,N]`` logits.

    The center payload remains BF16, INT8, or nibble-packed INT4 in memory;
    quantized values are unpacked and scaled only in registers.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not query.is_cuda:
        raise ValueError("Triton router requires CUDA tensors")
    if query.shape[-2] != 1:
        raise ValueError("packed Triton router supports decode q_len=1")
    if center_bits not in {4, 8, 16}:
        raise ValueError("center_bits must be 4, 8, or 16")
    if not 1 <= max_components <= 8:
        raise ValueError("max_components must lie in [1,8]")
    if center_bits < 16 and center_scale is None:
        raise ValueError("quantized compact centers require scales")
    if center_bits == 16:
        # The pointer is compile-time dead in this specialization, but Triton
        # still requires a valid tensor argument at launch.
        center_scale = alpha

    batch, heads, groups, _, dim = query.shape
    blocks = last_block - first_block
    if blocks <= 0:
        raise ValueError("empty block range")
    output = torch.empty(
        (batch, heads, groups, blocks), device=query.device, dtype=torch.float32
    )
    block_n = 16 if max_components > 2 else 32
    grid = (triton.cdiv(blocks, block_n), batch * heads * groups)
    _packed_lse_kernel[grid](
        query, centers, center_scale, log_counts, alpha,
        component_start, component_count, output,
        1.0 / math.sqrt(dim), first_block, blocks, dim, groups,
        query.stride(0), query.stride(1), query.stride(2), query.stride(4),
        centers.stride(0), centers.stride(1), centers.stride(2), centers.stride(3),
        center_scale.stride(0), center_scale.stride(1), center_scale.stride(2),
        log_counts.stride(0), log_counts.stride(1), log_counts.stride(2),
        alpha.stride(0), alpha.stride(1), alpha.stride(2),
        component_start.stride(0), component_start.stride(1), component_start.stride(2),
        component_count.stride(0), component_count.stride(1), component_count.stride(2),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        heads,
        CENTER_BITS=center_bits,
        MAX_COMPONENTS=max_components,
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=4,
    )
    return output
