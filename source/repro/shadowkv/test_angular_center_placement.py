"""Unit tests for the query-free angular center path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ShadowKV"))

from models.centroid_router_cache import (  # noqa: E402
    allocate_distortion_target_counts,
    allocate_rate_distortion_counts,
    demand_adaptive_rate_distortion_counts,
    hierarchical_log_rate_distortion_counts,
    evaluate_proxy_lse_gap_path,
    evaluate_self_lse_cvar_path,
    distortion_threshold_counts,
    fit_agglomerative_self_lse_paths,
    fit_angular_isolation_lse_paths,
    fit_residual_tail_path,
    fit_residual_isolation_lse_paths,
    fit_value_tail_path,
    rate_distortion_counts,
    priced_concave_marginal_counts,
    relative_rate_distortion_counts,
)


def test_causal_query_anchor_extends_instead_of_replacing_self_k_bank() -> None:
    generator = torch.Generator().manual_seed(0)
    keys = torch.randn(1, 1, 3, 8, 6, generator=generator)
    anchor = 3.0 * torch.randn(1, 1, 3, 1, 6, generator=generator)
    _, self_path = fit_agglomerative_self_lse_paths(
        keys, temperatures=(1.0,), cost_mode="max_gap"
    )
    risk, hybrid_path = fit_agglomerative_self_lse_paths(
        keys,
        temperatures=(1.0,),
        cost_mode="max_gap",
        extra_proxy_queries=anchor,
    )
    priced = evaluate_self_lse_cvar_path(
        keys,
        hybrid_path,
        temperatures=(1.0,),
        tail_fraction=0.25,
        extra_proxy_queries=anchor,
    )
    anchor_priced = evaluate_proxy_lse_gap_path(
        keys, hybrid_path, anchor
    )
    assert risk.shape == priced.shape == anchor_priced.shape == (1, 1, 3, 8)
    assert not torch.equal(self_path, hybrid_path)
    assert risk[..., 1:].le(risk[..., :-1] + 1e-6).all()
    assert priced[..., 1:].le(priced[..., :-1] + 1e-6).all()
    assert anchor_priced[..., 1:].le(anchor_priced[..., :-1] + 1e-6).all()
    assert (
        risk[..., -1].eq(0).all()
        and priced[..., -1].eq(0).all()
        and anchor_priced[..., -1].eq(0).all()
    )


def test_mean_relative_path_is_a_bounded_nested_distortion() -> None:
    generator = torch.Generator().manual_seed(31)
    keys = torch.randn(2, 3, 8, 16, generator=generator)
    risk, path = fit_agglomerative_self_lse_paths(
        keys, temperatures=(0.25,), cost_mode="mean_relative"
    )
    assert risk.shape == (2, 3, 8)
    assert path.shape == (2, 3, 8, 8)
    assert risk.ge(0).all() and risk.le(1).all()
    assert risk[..., 1:].le(risk[..., :-1] + 1e-6).all()
    assert torch.equal(risk[..., -1], torch.zeros_like(risk[..., -1]))


def test_rate_distortion_allocator_is_parallel_and_respects_cap() -> None:
    risk = torch.tensor([[0.80, 0.25, 0.00], [0.15, 0.10, 0.00]])
    counts, penalty = allocate_rate_distortion_counts(risk, 0.5)
    assert counts.sum().item() <= 3
    assert torch.equal(counts, rate_distortion_counts(risk, penalty))
    # The difficult first block receives the available upgrade.
    assert counts.tolist() == [2, 1]


def test_rate_distortion_allocator_can_buy_a_nonconcave_bundle() -> None:
    # The first block only becomes worthwhile after buying all three extra
    # centers.  Marginal cummin would hide the final 9.8 improvement, whereas
    # the complete-path Lagrangian can select the r=4 operating point directly.
    risk = torch.tensor([
        [10.0, 9.9, 9.8, 0.0],
        [1.0, 0.8, 0.7, 0.6],
        [1.0, 0.8, 0.7, 0.6],
        [1.0, 0.8, 0.7, 0.6],
    ])
    counts, _ = allocate_rate_distortion_counts(risk, 0.75)
    assert counts.tolist() == [4, 1, 1, 1]


def test_distortion_target_spends_only_on_the_hard_group() -> None:
    risk = torch.tensor([
        [[0.20, 0.05, 0.00], [0.10, 0.02, 0.00]],
        [[2.00, 0.80, 0.00], [1.60, 0.60, 0.00]],
    ])
    counts, price = allocate_distortion_target_counts(
        risk, torch.tensor([0.25, 0.70])
    )
    assert counts[0].tolist() == [1, 1]
    assert counts[1].sum().item() > 2
    selected = risk.gather(-1, (counts - 1).unsqueeze(-1)).squeeze(-1)
    assert selected[0].mean() <= 0.25
    assert selected[1].mean() <= 0.70
    assert torch.isfinite(price).all()


def test_demand_adaptive_price_preserves_hard_layer_and_compresses_easy_layer() -> None:
    # batch0 requests two centers per block at the conservative price; batch1
    # is already saturated at one and may safely use the higher price.
    risk = torch.tensor([
        [[[3.0, 0.0], [3.0, 0.0]]],
        [[[1.0, 0.0], [1.0, 0.0]]],
    ])
    counts, price, demand = demand_adaptive_rate_distortion_counts(
        risk, low_penalty=1.5, high_penalty=2.5, demand_threshold=1.5
    )
    assert demand.tolist() == [2.0, 1.0]
    assert counts[0].tolist() == [[2, 2]]
    assert counts[1].tolist() == [[1, 1]]
    assert price[:, 0].tolist() == [1.5, 2.5]


def test_hierarchical_log_objective_adapts_continuously_per_head() -> None:
    risk = torch.tensor([[[
        [0.20, 0.00, 0.00],
        [0.10, 0.00, 0.00],
    ], [
        [4.00, 1.00, 0.00],
        [3.50, 0.80, 0.00],
    ]]])
    counts, price, mean_extra = hierarchical_log_rate_distortion_counts(
        risk, alpha=3.0, beta=1.0, group_mode="head"
    )
    assert counts.shape == (1, 2, 2)
    assert counts[0, 0].tolist() == [1, 1]
    assert counts[0, 1].float().mean() > 1
    assert mean_extra[0, 0] == 0
    assert mean_extra[0, 1] > 0
    assert price[0, 0] > price[0, 1]


def test_hierarchical_log_layer_mode_shares_one_price_across_heads() -> None:
    risk = torch.tensor([[[
        [0.20, 0.00], [0.10, 0.00],
    ], [
        [4.00, 0.00], [3.50, 0.00],
    ]]])
    counts, price, mean_extra = hierarchical_log_rate_distortion_counts(
        risk, alpha=2.0, beta=1.0, group_mode="layer"
    )
    assert counts.shape == (1, 2, 2)
    assert price.shape == (1, 2)
    assert price[0, 0] == price[0, 1]
    assert mean_extra.shape == (1,)


def test_hierarchical_floor_never_prices_below_base() -> None:
    risk = torch.tensor([[ [
        [8.0, 6.0, 4.0, 2.0, 0.0],
        [7.0, 5.0, 3.0, 1.0, 0.0],
    ] ]])
    _, price, _ = hierarchical_log_rate_distortion_counts(
        risk,
        alpha=0.75,
        beta=0.5,
        group_mode="head",
        base_penalty=1.5,
    )
    assert torch.all(price >= 1.5)
    assert torch.all(price <= 3.0)


def test_relative_rate_distortion_has_no_global_quota() -> None:
    risk = torch.tensor([[
        [1.0, 0.95, 0.10, 0.0],
        [1.0, 0.20, 0.10, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]])
    counts = relative_rate_distortion_counts(risk, torch.tensor([0.20]))
    # Complete bundles are selected independently per block.  No target mean
    # number of centers is passed to this allocator.
    assert counts.tolist() == [[3, 2, 1]]


def test_priced_marginal_has_no_quota_and_enforces_precedence() -> None:
    risk = torch.tensor([[
        [1.0, 0.6, 0.1, 0.0],
        [1.0, 0.9, 0.0, 0.0],
    ]])
    counts = priced_concave_marginal_counts(risk, torch.tensor([0.2]))
    # First path has gains [0.4, 0.5, 0.1], concavified to [0.4, 0.4, 0.1].
    # Second has [0.1, 0.9, 0], concavified to [0.1, 0.1, 0].
    assert counts.tolist() == [[3, 1]]


def test_distortion_threshold_selects_first_feasible_order() -> None:
    risk = torch.tensor([[
        [1.0, 0.6, 0.1, 0.0],
        [0.4, 0.2, 0.1, 0.0],
    ]])
    counts = distortion_threshold_counts(risk, torch.tensor([0.3]))
    assert counts.tolist() == [[3, 2]]


def test_angular_path_isolates_least_coherent_key() -> None:
    keys = torch.tensor(
        [[[[1.0, 0.0], [0.9, 0.1], [0.8, -0.1], [-1.0, 0.0]]]]
    ).unsqueeze(0)
    risk, path = fit_angular_isolation_lse_paths(keys)
    assert risk.shape == (1, 1, 1, 2)
    assert path.shape == (1, 1, 1, 2, 4)
    assert path[..., 0, :].eq(0).all()
    assert path[..., 1, :].sum().item() == 1
    assert path[..., 1, 3].item() == 1
    assert risk[..., 1].le(risk[..., 0]).all()


def test_angular_path_is_scale_and_rotation_invariant() -> None:
    generator = torch.Generator().manual_seed(7)
    keys = torch.randn(2, 3, 5, 8, 16, generator=generator)
    rotation, _ = torch.linalg.qr(
        torch.randn(16, 16, generator=generator)
    )
    _, reference = fit_angular_isolation_lse_paths(keys)
    _, transformed = fit_angular_isolation_lse_paths(
        torch.matmul(keys, rotation) * 3.0
    )
    assert torch.equal(reference, transformed)


def test_residual_tail_isolates_farthest_key() -> None:
    keys = torch.tensor(
        [[[[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0], [3.0, 0.0]]]]
    ).unsqueeze(0)
    priority, path = fit_residual_tail_path(keys)
    assert priority.shape == (1, 1, 1)
    assert path.shape == (1, 1, 1, 2, 4)
    assert path[..., 0, :].eq(0).all()
    assert path[..., 1, 3].item() == 0
    assert path[..., 1, :3].eq(1).all()


def test_residual_tail_is_translation_rotation_and_scale_equivariant() -> None:
    generator = torch.Generator().manual_seed(19)
    keys = torch.randn(2, 3, 5, 8, 16, generator=generator)
    rotation, _ = torch.linalg.qr(
        torch.randn(16, 16, generator=generator)
    )
    offset = torch.randn(16, generator=generator)
    reference_priority, reference_path = fit_residual_tail_path(keys)
    priority, path = fit_residual_tail_path(
        torch.matmul(keys, rotation) * 3.0 + offset
    )
    assert torch.equal(reference_path, path)
    assert torch.allclose(priority, reference_priority * 3.0, rtol=1e-5, atol=1e-5)


def test_value_tail_allocates_by_k_and_isolates_v_outlier() -> None:
    keys = torch.tensor([[[0.0], [1.0], [2.0], [8.0]]])
    values = torch.tensor([[[0.0], [12.0], [0.0], [0.0]]])
    priority, path = fit_value_tail_path(keys, values)

    assert torch.allclose(priority, torch.tensor([5.25]))
    assert path[0, 0].tolist() == [0, 0, 0, 0]
    assert path[0, 1].tolist() == [1, 0, 1, 1]


def test_residual_self_path_isolates_farthest_key_and_reduces_risk() -> None:
    keys = torch.tensor(
        [[[[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0], [3.0, 0.0]]]]
    ).unsqueeze(0)
    risk, path = fit_residual_isolation_lse_paths(keys)
    assert risk.shape == (1, 1, 1, 2)
    assert path[..., 1, 3].item() == 0
    assert path[..., 1, :3].eq(1).all()
    assert risk[..., 1].le(risk[..., 0]).all()
