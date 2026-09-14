from __future__ import annotations

import pytest
import torch

from models.query_robust import (
    QueryRobustSummaryWorkspace,
    build_query_robust_page_summaries,
)


def _inputs(
    *, pages: int = 4, page_size: int = 5, heads: int = 2, dim: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260914 + pages)
    keys = torch.randn(pages, page_size, heads, dim, dtype=torch.bfloat16)
    vertices = torch.randn(heads, 7, dim, dtype=torch.bfloat16)
    return keys, vertices


def _assert_summary_close(left, right) -> None:
    for name in ("landmark", "bias", "epsilon", "dual", "gap", "errors"):
        torch.testing.assert_close(
            getattr(left, name), getattr(right, name), atol=2e-2, rtol=2e-2
        )


def test_summary_workspace_reuses_fixed_shape_buffers() -> None:
    keys, vertices = _inputs()
    reference = build_query_robust_page_summaries(
        keys,
        vertices,
        scale=0.5,
        solver_iters=8,
        solver_lr=0.25,
        uniform_p=False,
    )
    workspace = QueryRobustSummaryWorkspace(keys.device)
    first = workspace.build(
        keys,
        vertices,
        scale=0.5,
        solver_iters=8,
        solver_lr=0.25,
        uniform_p=False,
    )
    allocations_after_first = workspace.allocation_count
    second = workspace.build(
        keys,
        vertices,
        scale=0.5,
        solver_iters=8,
        solver_lr=0.25,
        uniform_p=False,
    )

    _assert_summary_close(first, reference)
    _assert_summary_close(second, reference)
    assert workspace.allocation_count == allocations_after_first
    assert bool(torch.isfinite(second.errors).all())


def test_summary_workspace_grows_for_new_page_shape() -> None:
    small_keys, vertices = _inputs(pages=2)
    large_keys, _ = _inputs(pages=6)
    workspace = QueryRobustSummaryWorkspace(small_keys.device)
    workspace.build(
        small_keys,
        vertices,
        scale=0.5,
        solver_iters=4,
        solver_lr=0.25,
        uniform_p=False,
    )
    allocations_after_small = workspace.allocation_count
    workspace.build(
        large_keys,
        vertices,
        scale=0.5,
        solver_iters=4,
        solver_lr=0.25,
        uniform_p=False,
    )
    assert workspace.allocation_count > allocations_after_small


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compiled QR backend requires CUDA"
)
def test_compiled_summary_matches_eager_summary() -> None:
    keys, vertices = _inputs()
    keys = keys.cuda()
    vertices = vertices.cuda()
    eager = QueryRobustSummaryWorkspace(keys.device, backend="eager")
    compiled = QueryRobustSummaryWorkspace(keys.device, backend="compile")

    eager_summary = eager.build(
        keys,
        vertices,
        scale=0.5,
        solver_iters=8,
        solver_lr=0.25,
        uniform_p=False,
    )
    compiled_summary = compiled.build(
        keys,
        vertices,
        scale=0.5,
        solver_iters=8,
        solver_lr=0.25,
        uniform_p=False,
    )

    _assert_summary_close(compiled_summary, eager_summary)
    assert bool(torch.isfinite(compiled_summary.errors).all())

