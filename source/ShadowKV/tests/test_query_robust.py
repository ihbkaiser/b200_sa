"""Contract tests for the Query-Robust reference implementation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.query_robust import (
    build_query_robust_page_summaries,
    load_query_robust_asset,
    score_query_robust_pages_reference,
    solve_query_robust_page,
)


def _mass(query: torch.Tensor, keys: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.logsumexp(scale * (query @ keys.transpose(0, 1)), dim=-1)


def _bound(
    query: torch.Tensor,
    landmark: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return scale * (query @ landmark.float()) + bias.float()


def test_solver_certifies_query_hull_after_bf16_landmark_storage():
    torch.manual_seed(7)
    keys = torch.randn(16, 8)
    vertices = torch.randn(12, 8)
    summary = solve_query_robust_page(
        keys, vertices, scale=8.0**-0.5, solver_iters=48, solver_lr=0.25
    )

    vertex_errors = _mass(vertices, keys, 8.0**-0.5) - _bound(
        vertices, summary.landmark, summary.bias, 8.0**-0.5
    )
    assert torch.all(vertex_errors >= -1e-5)
    assert torch.all(vertex_errors <= summary.epsilon + 1e-5)
    assert float(summary.dual) <= float(summary.epsilon) + 1e-5
    assert float(summary.gap) >= -1e-5
    torch.testing.assert_close(vertex_errors, summary.errors, atol=1e-5, rtol=1e-5)

    coefficients = torch.distributions.Dirichlet(
        torch.ones(vertices.shape[0])
    ).sample((64,))
    interior = coefficients @ vertices
    interior_errors = _mass(interior, keys, 8.0**-0.5) - _bound(
        interior, summary.landmark, summary.bias, 8.0**-0.5
    )
    assert torch.all(interior_errors <= summary.epsilon + 1e-5)


def test_solver_uniform_baseline_uses_mean_key_and_log_page_size():
    keys = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [-1.0, 5.0], [0.0, -2.0]]
    )
    vertices = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    summary = solve_query_robust_page(
        keys,
        vertices,
        scale=0.5,
        solver_iters=24,
        solver_lr=0.25,
        uniform_p=True,
    )
    assert torch.equal(summary.landmark, keys.mean(dim=0).to(torch.bfloat16))
    torch.testing.assert_close(summary.bias, torch.log(torch.tensor(4.0)))


def test_batched_summary_matches_single_page_reference():
    torch.manual_seed(19)
    keys = torch.randn(3, 5, 2, 4)
    vertices = torch.randn(2, 7, 4)
    batched = build_query_robust_page_summaries(
        keys,
        vertices,
        scale=0.5,
        solver_iters=12,
        solver_lr=0.25,
        uniform_p=False,
    )
    for page in range(3):
        for head in range(2):
            single = solve_query_robust_page(
                keys[page, :, head],
                vertices[head],
                scale=0.5,
                solver_iters=12,
                solver_lr=0.25,
            )
            assert torch.equal(batched.landmark[page, head], single.landmark)
            torch.testing.assert_close(batched.bias[page, head], single.bias)
            torch.testing.assert_close(batched.epsilon[page, head], single.epsilon)
            torch.testing.assert_close(batched.errors[page, head], single.errors)


def test_reference_scorer_uses_gqa_max_and_invalidates_stale_pages():
    query = torch.tensor([[[1.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.0, 0.5]]])
    landmark = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ],
        dtype=torch.bfloat16,
    )
    bias = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    epsilon = torch.tensor([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
    valid = torch.tensor([[True, True], [True, False], [True, True]])
    slots = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    scores = score_query_robust_pages_reference(
        query,
        landmark,
        bias,
        epsilon,
        valid,
        slots,
        scale=1.0,
        alpha=0.5,
    ).squeeze(0)
    expected = torch.tensor(
        [
            max(1.0 + 0.1 + 0.005, 1.0 + 0.2 + 0.01),
            float("inf"),
            max(2.0 + 0.5 + 0.025, 2.0 + 0.6 + 0.03),
        ]
    )
    assert torch.equal(torch.isinf(scores), torch.isinf(expected))
    torch.testing.assert_close(scores[~torch.isinf(scores)], expected[~torch.isinf(expected)])


def test_asset_loader_slices_tp_and_rejects_mismatched_partition(tmp_path):
    vertices = torch.arange(2 * 4 * 3 * 5, dtype=torch.float32).view(2, 4, 3, 5).to(torch.bfloat16)
    valid = torch.full((2, 4), 3, dtype=torch.int32)
    path = tmp_path / "vertices.pt"
    torch.save(
        {
            "vertices": vertices,
            "num_valid_vertices": valid,
            "meta": {"model_id": "model", "tp_world_size": 2},
        },
        path,
    )
    asset = load_query_robust_asset(
        path,
        num_layers=2,
        global_num_kv_heads=4,
        head_dim=5,
        num_vertices=3,
        tensor_parallel_rank=1,
        tensor_parallel_size=2,
        expected_model_id="/models/model",
    )
    assert tuple(asset.vertices.shape) == (2, 2, 3, 5)
    assert torch.equal(asset.vertices, vertices[:, 2:].float())

    with pytest.raises(ValueError, match="TP partition"):
        load_query_robust_asset(
            path,
            num_layers=2,
            global_num_kv_heads=4,
            head_dim=5,
            num_vertices=3,
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
        )


def test_asset_loader_rejects_fake_zero_padding(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "vertices": torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]], dtype=torch.bfloat16),
            "num_valid_vertices": torch.tensor([[2]], dtype=torch.int32),
            "meta": {"model_id": "model", "tp_world_size": 1},
        },
        path,
    )
    with pytest.raises(ValueError, match="padded"):
        load_query_robust_asset(
            path,
            num_layers=1,
            global_num_kv_heads=1,
            head_dim=2,
            num_vertices=3,
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
        )


def test_vendored_qwen3_asset_has_expected_checksum_and_contract():
    root = Path(__file__).parents[1] / "artifacts/query_robust/qwen3_4b_128k"
    path = root / "qwen3_4b_qr_vertices_m32_128k.pt"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "189b839536e53dac532b032504311b803438d0a968a5aed1db90223f0e76fd68"
    asset = load_query_robust_asset(
        path,
        num_layers=36,
        global_num_kv_heads=8,
        head_dim=128,
        num_vertices=32,
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        expected_model_id="Qwen3-4B-Instruct-2507",
        expected_model_fingerprint="cdbee75f17c01a7cc42f958dc650907174af0554",
        expected_rope_config={"rope_theta": 5000000.0, "rope_type": "default"},
    )
    assert tuple(asset.vertices.shape) == (36, 8, 32, 128)
    assert asset.vertices.dtype == torch.float32


def test_query_robust_cache_builds_and_selects_pages(tmp_path):
    from models.query_robust_cache import StreamingQueryRobustCache

    config = SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        head_dim=8,
    )
    vertices = torch.tensor(
        [[[[1.0] * 8, [-1.0] * 8]] * 2], dtype=torch.bfloat16
    )
    path = tmp_path / "tiny_vertices.pt"
    torch.save(
        {
            "vertices": vertices,
            "num_valid_vertices": torch.full((1, 2), 2, dtype=torch.int32),
            "meta": {"model_id": "tiny", "tp_world_size": 1},
        },
        path,
    )
    cache = StreamingQueryRobustCache(
        config,
        max_length=64,
        device="cpu",
        dtype=torch.float32,
        sparse_budget=16,
        block_size=8,
        dense_layers=0,
        prefix_tokens=0,
        recent_tokens=0,
        vertices_path=path,
        model_id="tiny",
        num_vertices=2,
    )
    keys = torch.zeros(1, 2, 32, 8)
    keys[:, :, :8] = 1.0
    values = torch.arange(keys.numel(), dtype=torch.float32).view_as(keys)
    cache.prefill_kv_cache(values, 0, keys)
    assert bool(cache.metadata_valid[0, :, :4].all())
    query = torch.ones(1, 4, 1, 8)
    selected = cache._select_block_ids(0, query)
    assert selected.shape == (1, 2, 2)
    assert torch.all((selected == 0).any(dim=-1))

    gathered_keys, gathered_values = cache.select_key_value_cache(0, query)
    offsets = torch.arange(8)
    expected_positions = (selected.unsqueeze(-1) * 8 + offsets).reshape(1, 2, -1)
    expected_keys = keys.gather(
        2, expected_positions.unsqueeze(-1).expand(-1, -1, -1, 8)
    )
    expected_values = values.gather(
        2, expected_positions.unsqueeze(-1).expand(-1, -1, -1, 8)
    )
    torch.testing.assert_close(gathered_keys, expected_keys)
    torch.testing.assert_close(gathered_values, expected_values)

    cache.clear()
    assert not bool(cache.metadata_valid.any())
