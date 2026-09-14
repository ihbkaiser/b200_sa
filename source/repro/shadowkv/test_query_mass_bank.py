"""Unit checks for the causal prompt-query bank used by qmass placement."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ShadowKV"))

from models.adaptive_centroid_streaming_cache import (  # noqa: E402
    StreamingAdaptiveCentroidLSECache,
)
from models.centroid_router_cache import (  # noqa: E402
    fit_observed_query_mass_paths,
)


def test_qmass_bank_samples_only_the_final_causal_window():
    cache = StreamingAdaptiveCentroidLSECache.__new__(
        StreamingAdaptiveCentroidLSECache
    )
    cache.center_placement = "qmass_bank"
    cache.query_mass_bank_size = 4
    cache.query_mass_window = 6
    query = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1)
    bank = cache.prepare_prefill_query(query)
    assert bank.flatten().tolist() == [4.0, 6.0, 7.0, 9.0]


def test_other_placements_keep_only_the_final_query():
    cache = StreamingAdaptiveCentroidLSECache.__new__(
        StreamingAdaptiveCentroidLSECache
    )
    cache.center_placement = "qmass_one"
    query = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1)
    assert cache.prepare_prefill_query(query).flatten().tolist() == [4.0]


def test_topk_hinge_prices_only_unsafe_exact_shortlist_blocks():
    keys = torch.tensor([
        [[[[4.0, 0.0], [0.0, 0.0]], [[-1.0, 0.0], [-1.0, 0.0]]]]
    ])
    query = torch.tensor([[[[1.0, 0.0]]]])
    risk, _, _ = fit_observed_query_mass_paths(
        keys,
        query,
        objective="topk_hinge",
        selection_fraction=0.5,
    )
    assert risk[0, 0, 0, 0] > 0
    assert risk[0, 0, 1, 0] == 0
    assert torch.equal(risk[..., 1], torch.zeros_like(risk[..., 1]))


def test_topk_hinge_cvar_prices_the_worst_observed_query():
    keys = torch.tensor([
        [[[[4.0, 0.0], [0.0, 0.0]], [[0.0, 4.0], [0.0, 0.0]]]]
    ])
    query = torch.tensor([[[[1.0, 0.0], [0.0, 0.2]]]])
    mean_risk, _, _ = fit_observed_query_mass_paths(
        keys, query, objective="topk_hinge", selection_fraction=0.5,
    )
    worst_risk, _, _ = fit_observed_query_mass_paths(
        keys,
        query,
        objective="topk_hinge_cvar",
        tail_fraction=0.5,
        selection_fraction=0.5,
    )
    assert torch.all(worst_risk[..., :-1] >= mean_risk[..., :-1])
    assert torch.any(worst_risk[..., :-1] > mean_risk[..., :-1])
