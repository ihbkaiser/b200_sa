from __future__ import annotations

import importlib.util

import pytest
import torch

import models.query_robust_cache as query_robust_cache_module
from models.query_robust import score_query_robust_pages_reference
from models.query_robust_cache import StreamingQueryRobustCache
from models.query_robust_triton import query_robust_page_scores


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="QR Triton parity tests require CUDA and Triton",
)


def _fixture() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260914)
    device = torch.device("cuda")
    batch, query_heads, kv_heads, pages, dim = 1, 32, 8, 96, 128
    query = torch.randn(
        batch, query_heads, dim, device=device, dtype=torch.bfloat16
    )
    landmark = torch.randn(pages, kv_heads, dim, device=device, dtype=torch.bfloat16)
    bias = torch.randn(pages, kv_heads, device=device, dtype=torch.float32)
    epsilon = torch.rand(pages, kv_heads, device=device, dtype=torch.float32)
    valid = torch.ones(pages, kv_heads, device=device, dtype=torch.bool)
    return query, landmark, bias, epsilon, valid


def test_fused_qr_scores_match_reference_and_topk() -> None:
    query, landmark, bias, epsilon, valid = _fixture()
    first_page, last_page = 7, 71
    slots = torch.arange(
        first_page, last_page, device=query.device, dtype=torch.int32
    ).view(1, -1)

    reference = score_query_robust_pages_reference(
        query,
        landmark,
        bias,
        epsilon,
        valid,
        slots,
        scale=0.08838834764831843,
        alpha=1.75,
    )
    fused = query_robust_page_scores(
        query,
        landmark,
        bias,
        epsilon,
        valid,
        first_page=first_page,
        last_page=last_page,
        scale=0.08838834764831843,
        alpha=1.75,
    )

    assert fused.shape == (1, last_page - first_page)
    assert fused.dtype == torch.float32
    torch.testing.assert_close(fused, reference, atol=2e-2, rtol=2e-2)
    assert torch.equal(torch.isinf(fused), torch.isinf(reference))
    topk = min(32, fused.shape[-1])
    assert torch.equal(
        torch.topk(fused, topk, dim=-1).indices,
        torch.topk(reference, topk, dim=-1).indices,
    )


def test_fused_qr_preserves_invalid_page_semantics() -> None:
    query, landmark, bias, epsilon, valid = _fixture()
    first_page, last_page = 3, 67
    invalid_page = 19
    valid[invalid_page, 2] = False
    slots = torch.arange(
        first_page, last_page, device=query.device, dtype=torch.int32
    ).view(1, -1)

    reference = score_query_robust_pages_reference(
        query,
        landmark,
        bias,
        epsilon,
        valid,
        slots,
        scale=0.125,
        alpha=0.5,
    )
    fused = query_robust_page_scores(
        query,
        landmark,
        bias,
        epsilon,
        valid,
        first_page=first_page,
        last_page=last_page,
        scale=0.125,
        alpha=0.5,
    )

    relative = invalid_page - first_page
    assert torch.isinf(fused[0, relative])
    assert torch.equal(torch.isinf(fused), torch.isinf(reference))
    finite = torch.isfinite(reference)
    torch.testing.assert_close(fused[finite], reference[finite], atol=2e-2, rtol=2e-2)


def test_fused_qr_honors_independent_metadata_strides() -> None:
    query, landmark, bias, epsilon, valid = _fixture()
    first_page, last_page = 5, 69
    epsilon = torch.rand(
        epsilon.shape[1], epsilon.shape[0], device=epsilon.device, dtype=epsilon.dtype
    ).transpose(0, 1)
    assert not epsilon.is_contiguous()
    slots = torch.arange(
        first_page, last_page, device=query.device, dtype=torch.int32
    ).view(1, -1)

    reference = score_query_robust_pages_reference(
        query,
        landmark,
        bias,
        epsilon,
        valid,
        slots,
        scale=0.08838834764831843,
        alpha=1.25,
    )
    fused = query_robust_page_scores(
        query,
        landmark,
        bias,
        epsilon,
        valid,
        first_page=first_page,
        last_page=last_page,
        scale=0.08838834764831843,
        alpha=1.25,
    )
    torch.testing.assert_close(fused, reference, atol=2e-2, rtol=2e-2)


def test_qr_cache_dispatches_contiguous_cuda_decode_to_triton(monkeypatch) -> None:
    query, landmark, bias, epsilon, valid = _fixture()
    cache = StreamingQueryRobustCache.__new__(StreamingQueryRobustCache)
    cache.batch_size = 1
    cache.num_attention_heads = 32
    cache.num_key_value_heads = 8
    cache.num_key_value_groups = 4
    cache.head_dim = 128
    cache.compute_device = query.device
    cache.group_reduce = "max"
    cache.incoming_q_len = 1
    cache.query_robust_scale = 0.08838834764831843
    cache.score_alpha = 1.0
    cache.query_robust_router_backend = "triton"
    cache.landmark_cache = [landmark.permute(1, 0, 2).contiguous()]
    cache.bias_cache = [bias.transpose(0, 1).contiguous()]
    cache.epsilon_cache = [epsilon.transpose(0, 1).contiguous()]
    cache.metadata_valid = [valid.transpose(0, 1).contiguous()]

    called = {"fused": False}
    real_fused = query_robust_page_scores

    def wrapped_fused(*args, **kwargs):
        called["fused"] = True
        return real_fused(*args, **kwargs)

    monkeypatch.setattr(
        query_robust_cache_module,
        "query_robust_page_scores",
        wrapped_fused,
        raising=False,
    )
    query_states = query.view(1, 32, 1, 128)
    scores = cache._score_blocks(0, query_states, 7, 71)

    assert called["fused"]
    assert scores.shape == (1, 8, 64)


@pytest.mark.parametrize("bad_range", [(0, 0), (10, 4)])
def test_fused_qr_rejects_empty_page_ranges(bad_range: tuple[int, int]) -> None:
    query, landmark, bias, epsilon, valid = _fixture()
    with pytest.raises(ValueError, match="page range"):
        query_robust_page_scores(
            query,
            landmark,
            bias,
            epsilon,
            valid,
            first_page=bad_range[0],
            last_page=bad_range[1],
            scale=0.125,
            alpha=1.0,
        )
