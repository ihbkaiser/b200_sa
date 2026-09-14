import math

import pytest
import torch

from models.centroid_router_cache import (
    allocate_concave_marginal_counts,
    allocate_minimax_centroid_counts,
    fit_block_centroids,
    fit_agglomerative_self_lse_paths,
    fit_direct_lse_two_centroids,
    fit_exact_key_only_r_centroids,
    fit_key_only_two_centroids,
    fit_lazy_exact_self_lse_allocation,
    fit_minimax_two_centroids,
    fit_residual_tail_lse_paths,
    fit_self_lse_r_centroids,
    fit_self_lse_two_centroids,
    pack_adaptive_centroids,
)


@pytest.mark.parametrize(
    "method,r",
    [("farthest_body", 2), ("kmeans", 2), ("kmeans", 4), ("minimax2", 2)],
)
def test_centroid_mixture_is_a_jensen_lower_bound(method, r):
    torch.manual_seed(11)
    keys = torch.randn(2, 3, 5, 8, 6)
    query = torch.randn(2, 3, 5, 6)
    centers, counts = fit_block_centroids(keys, r, method)
    approximate = torch.logsumexp(
        torch.einsum("...d,...rd->...r", query, centers)
        + counts.float().log(),
        dim=-1,
    )
    exact = torch.logsumexp(
        torch.einsum("...d,...sd->...s", query, keys), dim=-1
    )
    assert torch.all(approximate <= exact + 2e-5)
    torch.testing.assert_close(
        torch.einsum("...r,...rd->...d", counts.to(centers.dtype), centers),
        keys.sum(dim=-2),
        rtol=2e-5,
        atol=2e-5,
    )


def test_eight_centroids_recover_exact_logsumexp():
    torch.manual_seed(19)
    keys = torch.randn(2, 4, 8, 7)
    query = torch.randn(2, 4, 7)
    centers, counts = fit_block_centroids(keys, 8, "kmeans")
    approximate = torch.logsumexp(
        torch.einsum("...d,...rd->...r", query, centers)
        + counts.float().log(),
        dim=-1,
    )
    exact = torch.logsumexp(
        torch.einsum("...d,...sd->...s", query, keys), dim=-1
    )
    torch.testing.assert_close(approximate, exact, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("objective", ["scatter", "cosine", "minimax"])
@pytest.mark.parametrize("r", [1, 2, 3, 4, 8])
def test_exact_r_centroids_preserve_sum_and_lower_bound(objective, r):
    torch.manual_seed(20 + r)
    keys = torch.randn(2, 8, 5)
    query = torch.randn(2, 5)
    centers, counts, _ = fit_exact_key_only_r_centroids(keys, r, objective)
    torch.testing.assert_close(
        torch.einsum("...r,...rd->...d", counts.to(centers.dtype), centers),
        keys.sum(-2),
        rtol=2e-5,
        atol=2e-5,
    )
    approximate = torch.logsumexp(
        torch.einsum("...d,...rd->...r", query, centers) + counts.float().log(),
        dim=-1,
    )
    exact = torch.logsumexp(torch.einsum("...d,...sd->...s", query, keys), dim=-1)
    assert torch.all(approximate <= exact + 2e-5)
    if r == 8:
        torch.testing.assert_close(approximate, exact, rtol=2e-5, atol=2e-5)


def test_exact_r_scatter_matches_separated_three_clusters():
    keys = torch.tensor([[[0.0], [0.2], [4.0], [4.3], [9.0], [9.1]]])
    centers, counts, objective = fit_exact_key_only_r_centroids(keys, 3, "scatter")
    torch.testing.assert_close(
        centers[0, :, 0].sort().values,
        torch.tensor([0.1, 4.15, 9.05]),
        atol=1e-5,
        rtol=0,
    )
    assert counts[0].sort().values.tolist() == [2, 2, 2]
    assert objective.item() < 0.001


@pytest.mark.parametrize("r", [1, 2, 3, 4, 8])
def test_self_lse_r_centroids_preserve_sum_and_lower_bound(r):
    torch.manual_seed(73 + r)
    keys = torch.randn(3, 8, 6)
    query = torch.randn(3, 6)
    centers, counts, risk = fit_self_lse_r_centroids(keys, r)
    assert torch.all(risk >= 0)
    torch.testing.assert_close(
        torch.einsum("...r,...rd->...d", counts.to(centers.dtype), centers),
        keys.sum(-2),
        rtol=2e-5,
        atol=2e-5,
    )
    approximate = torch.logsumexp(
        torch.einsum("...d,...rd->...r", query, centers) + counts.float().log(),
        dim=-1,
    )
    exact = torch.logsumexp(torch.einsum("...d,...sd->...s", query, keys), dim=-1)
    assert torch.all(approximate <= exact + 2e-5)
    if r == 8:
        torch.testing.assert_close(approximate, exact, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(risk, torch.zeros_like(risk), atol=2e-5, rtol=0)


def test_self_lse_two_centroids_reports_nonnegative_gain():
    torch.manual_seed(91)
    keys = torch.randn(7, 8, 5)
    centers, counts, risk_one, risk_two, gain = fit_self_lse_two_centroids(keys)
    assert centers.shape == (7, 2, 5)
    assert counts.shape == (7, 2)
    assert torch.all(risk_two <= risk_one + 2e-5)
    torch.testing.assert_close(gain, risk_one - risk_two, atol=2e-5, rtol=2e-5)


def test_self_lse_two_centroids_returns_isotropic_component_correction():
    torch.manual_seed(92)
    keys = torch.randn(7, 8, 5)
    centers, counts, _, _, _, alpha = fit_self_lse_two_centroids(
        keys, return_isotropic_alpha=True
    )
    assert centers.shape == (7, 2, 5)
    assert counts.shape == (7, 2)
    assert alpha.shape == (7, 2)
    assert torch.all(torch.isfinite(alpha))
    assert torch.all(alpha >= 0)


def test_agglomerative_self_lse_path_is_nested_monotone_and_exact_at_eight():
    torch.manual_seed(193)
    keys = torch.randn(5, 8, 7)
    risk, assignment = fit_agglomerative_self_lse_paths(
        keys, temperatures=(1.0, 1.5)
    )
    assert risk.shape == (5, 8)
    assert assignment.shape == (5, 8, 8)
    assert torch.all(risk[:, 1:] <= risk[:, :-1] + 2e-5)
    torch.testing.assert_close(risk[:, -1], torch.zeros(5), atol=2e-5, rtol=0)
    for r in range(1, 9):
        assert torch.all(assignment[:, r - 1].amax(-1) < r)
        assert torch.all(
            torch.nn.functional.one_hot(
                assignment[:, r - 1].long(), num_classes=r
            ).sum(-2)
            > 0
        )
    # Every finer assignment must refine the immediately coarser one.
    for r in range(1, 8):
        coarse = assignment[:, r - 1]
        fine = assignment[:, r]
        for block in range(keys.shape[0]):
            for label in range(r + 1):
                members = coarse[block, fine[block] == label]
                assert members.unique().numel() <= 1


def test_relative_mass_path_is_bounded_monotone_and_exact_at_eight():
    torch.manual_seed(1943)
    keys = torch.randn(6, 8, 9)
    soft, assignment = fit_agglomerative_self_lse_paths(
        keys,
        temperatures=(1.0, 1.5),
        cost_mode="relative_lme",
        cost_beta=4.0,
    )
    hard, _ = fit_agglomerative_self_lse_paths(
        keys,
        temperatures=(1.0, 1.5),
        cost_mode="relative_lme",
        cost_beta=16.0,
    )
    assert assignment.shape == (6, 8, 8)
    assert torch.all((soft >= 0) & (soft <= 1 + 2e-5))
    assert torch.all(soft[:, 1:] <= soft[:, :-1] + 2e-5)
    torch.testing.assert_close(soft[:, -1], torch.zeros(6), atol=2e-5, rtol=0)
    # Entropic risk moves toward the worst proxy direction as beta grows.
    assert torch.all(hard >= soft - 2e-5)


@pytest.mark.parametrize(
    "mode",
    ["mean_gap", "gap_lme", "weighted_gap_lme", "weighted_relative_lme"],
)
def test_smooth_self_lse_cost_paths_are_monotone_and_exact(mode):
    torch.manual_seed(1944)
    keys = torch.randn(5, 8, 7)
    risk, assignment = fit_agglomerative_self_lse_paths(
        keys,
        temperatures=(1.0,),
        cost_mode=mode,
        cost_beta=4.0,
    )
    assert assignment.shape == (5, 8, 8)
    assert torch.all(risk >= 0)
    assert torch.all(risk[:, 1:] <= risk[:, :-1] + 2e-5)
    torch.testing.assert_close(risk[:, -1], torch.zeros(5), atol=2e-5, rtol=0)


def test_marginal_allocator_can_spend_multiple_centers_on_one_block():
    risk = torch.tensor(
        [
            [1.0, 0.5, 0.1, 0.0],
            [0.3, 0.2, 0.1, 0.0],
        ]
    )
    counts = allocate_concave_marginal_counts(risk, extra_fraction=1.0)
    assert counts.tolist() == [3, 1]
    assert counts.sum().item() == 4


@pytest.mark.parametrize("selection_mode", ["residual", "angular", "norm"])
def test_residual_tail_path_is_nested_monotone_and_exact_at_eight(selection_mode):
    torch.manual_seed(1945)
    keys = torch.randn(5, 8, 7)
    risk, assignment = fit_residual_tail_lse_paths(
        keys, selection_mode=selection_mode
    )
    assert risk.shape == (5, 8)
    assert assignment.shape == (5, 8, 8)
    assert torch.all(risk[:, 1:] <= risk[:, :-1] + 2e-5)
    torch.testing.assert_close(risk[:, -1], torch.zeros(5), atol=2e-5, rtol=0)
    for r in range(1, 9):
        counts = torch.nn.functional.one_hot(
            assignment[:, r - 1].long(), num_classes=r
        ).sum(-2)
        assert torch.all(counts > 0)
        if r > 1:
            # All tail components are exact singletons; component zero is the
            # only body that can contain more than one token.
            assert torch.all(counts[:, 1:] == 1)


def test_minimax_allocator_crosses_a_small_first_gain_to_reach_later_drop():
    # Block 0 needs two prerequisite upgrades to expose its large reduction;
    # marginal-gain allocation would prefer block 1 after the first step.
    risk = torch.tensor([[10.0, 9.0, 0.0], [8.0, 0.0, 0.0]])
    counts = allocate_minimax_centroid_counts(risk, extra_fraction=1.0)
    assert counts.tolist() == [3, 1]
    assert counts.sum().item() == 4


def test_minimax_allocator_supports_more_than_one_extra_per_block():
    risk = torch.tensor(
        [
            [9.0, 7.0, 5.0, 3.0, 0.0],
            [8.0, 6.0, 4.0, 2.0, 0.0],
        ]
    )
    counts = allocate_minimax_centroid_counts(risk, extra_fraction=3.0)
    assert counts.sum().item() == 8
    assert torch.all((counts >= 1) & (counts <= 5))


def test_packed_adaptive_centroids_preserve_weighted_key_sums():
    torch.manual_seed(194)
    keys = torch.randn(2, 3, 6, 8, 5)
    risk, assignment = fit_agglomerative_self_lse_paths(
        keys, temperatures=(1.0,)
    )
    component_counts = allocate_minimax_centroid_counts(risk, 0.5)
    centers, populations, alpha, block_ids = pack_adaptive_centroids(
        keys, assignment, component_counts
    )
    assert centers.shape[-2] == 9
    assert torch.all(populations > 0)
    assert torch.all(alpha >= 0)
    reconstructed = torch.zeros_like(keys.sum(-2))
    reconstructed.scatter_add_(
        -2,
        block_ids[..., None].expand(*block_ids.shape, keys.shape[-1]),
        centers * populations[..., None],
    )
    torch.testing.assert_close(reconstructed, keys.sum(-2), atol=2e-5, rtol=2e-5)


def test_lazy_exact_allocation_matches_bruteforce_minimax():
    torch.manual_seed(195)
    keys = torch.randn(2, 4, 5)
    risk, assignment, counts = fit_lazy_exact_self_lse_allocation(
        keys, extra_fraction=1.0, temperatures=(1.0,)
    )
    exact_risk = []
    for r in range(1, 5):
        _, _, value = fit_self_lse_r_centroids(
            keys, r, temperatures=(1.0,)
        )
        exact_risk.append(value)
    exact_risk = torch.stack(exact_risk, dim=-1)
    brute = float("inf")
    for first in range(1, 5):
        for second in range(1, 5):
            if first + second <= 4:
                brute = min(
                    brute,
                    max(
                        exact_risk[0, first - 1].item(),
                        exact_risk[1, second - 1].item(),
                    ),
                )
    achieved = max(
        exact_risk[0, counts[0] - 1].item(),
        exact_risk[1, counts[1] - 1].item(),
    )
    assert achieved == pytest.approx(brute, abs=2e-5)
    assert counts.sum().item() == 4
    for block in range(2):
        r = counts[block].item()
        assert assignment[block, r - 1].amax().item() < r
        assert risk[block, r - 1].item() == pytest.approx(
            exact_risk[block, r - 1].item(), abs=2e-5
        )


def test_minimax_two_centroids_finds_separated_pairs():
    keys = torch.tensor([[[0.0], [0.2], [10.0], [10.2]]])
    centers, counts, radius_one, radius_two = fit_minimax_two_centroids(keys)
    torch.testing.assert_close(
        centers[0, :, 0].sort().values,
        torch.tensor([0.1, 10.1]),
        atol=1e-5,
        rtol=0,
    )
    assert counts[0].sort().values.tolist() == [2, 2]
    torch.testing.assert_close(radius_one, torch.tensor([5.1]), atol=1e-5, rtol=0)
    torch.testing.assert_close(radius_two, torch.tensor([0.1]), atol=1e-4, rtol=0)


def test_minimax_two_centroids_matches_bruteforce_objective():
    torch.manual_seed(23)
    keys = torch.randn(3, 8, 5)
    _, _, _, radius_two = fit_minimax_two_centroids(keys)
    brute = []
    for block in keys:
        best = float("inf")
        for bits in range(2 ** 7 - 1):
            mask = torch.tensor([True] + [bool((bits >> j) & 1) for j in range(7)])
            a, b = block[mask], block[~mask]
            ra = (a - a.mean(0)).norm(dim=-1).max()
            rb = (b - b.mean(0)).norm(dim=-1).max()
            best = min(best, max(ra.item(), rb.item()))
        brute.append(best)
    torch.testing.assert_close(radius_two, torch.tensor(brute), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("objective", ["scatter", "cosine"])
def test_key_only_two_centroids_preserve_sum_and_reduce_dispersion(objective):
    keys = torch.tensor(
        [[[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]]]
    )
    centers, counts, one, two, gain = fit_key_only_two_centroids(keys, objective)
    assert counts[0].sort().values.tolist() == [2, 2]
    assert two.item() < one.item()
    torch.testing.assert_close(gain, one - two)
    torch.testing.assert_close(
        torch.einsum("...r,...rd->...d", counts.to(centers.dtype), centers),
        keys.sum(dim=-2),
    )


@pytest.mark.parametrize("objective", ["scatter", "cosine"])
def test_key_only_two_centroid_lse_is_jensen_lower_bound(objective):
    torch.manual_seed(29)
    keys = torch.randn(5, 8, 7)
    query = torch.randn(5, 7)
    centers, counts, *_ = fit_key_only_two_centroids(keys, objective)
    approximate = torch.logsumexp(
        torch.einsum("...d,...rd->...r", query, centers)
        + counts.float().log(),
        dim=-1,
    )
    exact = torch.logsumexp(
        torch.einsum("...d,...sd->...s", query, keys), dim=-1
    )
    assert torch.all(approximate <= exact + 2e-5)


def test_direct_lse_optimizer_reduces_calibrated_risk_on_two_clusters():
    keys = torch.tensor(
        [[[0.0], [0.2], [10.0], [10.2]], [[-2.0], [-1.8], [3.0], [3.3]]]
    )
    queries = torch.tensor([[-1.0], [-0.4], [0.3], [0.8], [1.2]])
    (
        centers,
        counts,
        alpha_one,
        alpha_two,
        risk_one,
        risk_two,
        diameter_one,
        diameter_two,
    ) = fit_direct_lse_two_centroids(keys, queries)
    assert centers.shape == (2, 2, 1)
    assert counts.shape == (2, 2)
    assert torch.all(counts.sum(-1) == 4)
    assert torch.all(alpha_one >= 0) and torch.all(alpha_two >= 0)
    assert torch.all(risk_two < risk_one)
    assert torch.all(diameter_two.amax(-1) < diameter_one)
