"""Pinned-CPU KV gather used by the streaming retrieval caches.

The fast path is a small, independently implemented CUDA/UVA kernel.  It
reads selected rows directly from pinned host memory and writes K and V into
GPU attention buffers in one launch.  A pure PyTorch fallback is kept for
portability and correctness testing.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_uva_extension():
    from torch.utils.cpp_extension import load

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = os.path.join(root, "kernels", "streaming_uva_gather.cu")
    return load(
        name="shadowkv_streaming_uva_gather",
        sources=[source],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=os.environ.get("SHADOWKV_VERBOSE_BUILD") == "1",
    )


def _validate(
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    position_ids: torch.Tensor,
    destination_k: torch.Tensor,
    destination_v: torch.Tensor,
) -> int:
    if source_k.shape != source_v.shape or source_k.ndim != 4:
        raise ValueError("CPU K/V sources must have the same [B,H,N,D] shape")
    if destination_k.shape != destination_v.shape or destination_k.ndim != 4:
        raise ValueError("GPU K/V destinations must have the same [B,H,N,D] shape")
    if position_ids.shape[:2] != source_k.shape[:2] or position_ids.ndim != 3:
        raise ValueError("position_ids must have shape [B,H,K]")
    count = position_ids.shape[-1]
    if destination_k.shape[:2] != source_k.shape[:2]:
        raise ValueError("source and destination B/H dimensions differ")
    if destination_k.shape[-1] != source_k.shape[-1]:
        raise ValueError("source and destination head dimensions differ")
    if count > destination_k.shape[-2]:
        raise ValueError("destination capacity is smaller than selected count")
    if source_k.dtype != source_v.dtype or source_k.dtype != destination_k.dtype:
        raise ValueError("source and destination dtypes differ")
    if destination_k.dtype != destination_v.dtype:
        raise ValueError("destination K/V dtypes differ")
    return count


@torch.inference_mode()
def gather_kv_torch(
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    position_ids: torch.Tensor,
    destination_k: torch.Tensor,
    destination_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Portable pinned-CPU gather; correct but synchronizes through CPU ids."""
    count = _validate(
        source_k, source_v, position_ids, destination_k, destination_v
    )
    cpu_ids = position_ids.to(device="cpu")
    for batch in range(source_k.shape[0]):
        for head in range(source_k.shape[1]):
            ids = cpu_ids[batch, head]
            destination_k[batch, head, :count].copy_(
                source_k[batch, head].index_select(0, ids), non_blocking=True
            )
            destination_v[batch, head, :count].copy_(
                source_v[batch, head].index_select(0, ids), non_blocking=True
            )
    return destination_k[..., :count, :], destination_v[..., :count, :]


@torch.inference_mode()
def gather_kv_uva(
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    position_ids: torch.Tensor,
    destination_k: torch.Tensor,
    destination_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V directly from pinned host memory with a CUDA UVA kernel."""
    count = _validate(
        source_k, source_v, position_ids, destination_k, destination_v
    )
    if not source_k.is_cpu or not source_v.is_cpu:
        raise ValueError("UVA sources must be CPU tensors")
    if not source_k.is_pinned() or not source_v.is_pinned():
        raise ValueError("UVA sources must use pinned memory")
    if not destination_k.is_cuda or not destination_v.is_cuda:
        raise ValueError("UVA destinations must be CUDA tensors")
    if not position_ids.is_cuda or position_ids.dtype != torch.int64:
        raise ValueError("UVA position_ids must be CUDA int64")
    for tensor in (source_k, source_v, destination_k, destination_v, position_ids):
        if not tensor.is_contiguous():
            raise ValueError("UVA gather requires contiguous tensors")
    _load_uva_extension().gather_kv(
        source_k,
        source_v,
        position_ids,
        destination_k,
        destination_v,
    )
    return destination_k[..., :count, :], destination_v[..., :count, :]


@torch.inference_mode()
def gather_kv(
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    position_ids: torch.Tensor,
    destination_k: torch.Tensor,
    destination_v: torch.Tensor,
    *,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch an offloaded KV gather.

    ``auto`` tries the CUDA/UVA implementation and falls back to the portable
    implementation if compilation is unavailable.  ``uva`` is strict and is
    intended for performance/correctness gates.
    """
    if backend not in {"auto", "uva", "torch"}:
        raise ValueError("offload gather backend must be auto, uva, or torch")
    if backend == "torch":
        return gather_kv_torch(
            source_k, source_v, position_ids, destination_k, destination_v
        )
    try:
        return gather_kv_uva(
            source_k, source_v, position_ids, destination_k, destination_v
        )
    except Exception:
        if backend == "uva":
            raise
        return gather_kv_torch(
            source_k, source_v, position_ids, destination_k, destination_v
        )


@torch.inference_mode()
def append_kv_uva(
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    destination_k: torch.Tensor,
    destination_v: torch.Tensor,
    staging_k: torch.Tensor,
    *,
    destination_start: int,
    staging_start: int,
) -> None:
    """Append a small decode batch to pinned K/V and GPU block staging."""
    if source_k.shape != source_v.shape or source_k.ndim != 4:
        raise ValueError("append sources must have matching [B,H,Q,D] shapes")
    if not source_k.is_cuda or not source_v.is_cuda or not staging_k.is_cuda:
        raise ValueError("append sources and staging must be CUDA tensors")
    if not destination_k.is_cpu or not destination_v.is_cpu:
        raise ValueError("append destinations must be CPU tensors")
    if not destination_k.is_pinned() or not destination_v.is_pinned():
        raise ValueError("append destinations must use pinned memory")
    for tensor in (source_k, source_v, destination_k, destination_v, staging_k):
        if not tensor.is_contiguous():
            raise ValueError("UVA append requires contiguous tensors")
    _load_uva_extension().append_kv(
        source_k,
        source_v,
        destination_k,
        destination_v,
        staging_k,
        destination_start,
        staging_start,
    )


@torch.inference_mode()
def gather_blocks_reuse_uva(
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    previous_k: torch.Tensor,
    previous_v: torch.Tensor,
    previous_blocks: torch.Tensor,
    current_blocks: torch.Tensor,
    destination_k: torch.Tensor,
    destination_v: torch.Tensor,
    destination_blocks: torch.Tensor,
    *,
    block_size: int,
    exact_ranges: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse unchanged selected blocks and fetch only replacements via UVA.

    The destination follows ``current_blocks`` order exactly.  At most two
    exact prefix/suffix ranges are appended after the dynamically selected
    blocks, matching :class:`StreamingBlockState`.
    """
    if len(exact_ranges) > 2:
        raise ValueError("reuse gather supports at most two exact ranges")
    ranges = list(exact_ranges) + [(0, 0)] * (2 - len(exact_ranges))
    (start0, end0), (start1, end1) = ranges
    count = current_blocks.shape[-1] * block_size + (end0 - start0) + (
        end1 - start1
    )
    _load_uva_extension().gather_blocks_reuse_kv(
        source_k,
        source_v,
        previous_k,
        previous_v,
        previous_blocks,
        current_blocks,
        destination_k,
        destination_v,
        destination_blocks,
        block_size,
        start0,
        end0 - start0,
        start1,
        end1 - start1,
    )
    return destination_k[..., :count, :], destination_v[..., :count, :]


@torch.inference_mode()
def gather_blocks_reuse_values_uva(
    source_v: torch.Tensor,
    previous_v: torch.Tensor,
    previous_blocks: torch.Tensor,
    current_blocks: torch.Tensor,
    destination_v: torch.Tensor,
    destination_blocks: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """Block gather with reuse for a cache that moves only values.

    ShadowKV keeps keys as a low-rank factorisation on the GPU, so pulling K
    across PCIe alongside V would double its traffic for bytes it discards.
    """
    count = current_blocks.shape[-1] * block_size
    _load_uva_extension().gather_blocks_reuse_values(
        source_v,
        previous_v,
        previous_blocks,
        current_blocks,
        destination_v,
        destination_blocks,
        block_size,
    )
    return destination_v[..., :count, :]
