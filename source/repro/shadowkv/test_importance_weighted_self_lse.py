import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ShadowKV"))

from models.centroid_router_cache import fit_agglomerative_self_lse_paths


def test_importance_weighted_paths_cover_orders_one_to_eight():
    generator = torch.Generator().manual_seed(7)
    keys = torch.randn(1, 2, 5, 8, 16, generator=generator)
    weight = torch.rand(1, 2, 5, 8, generator=generator) + 0.1
    risk, path = fit_agglomerative_self_lse_paths(
        keys,
        temperatures=(1.0,),
        cost_mode="importance_relative",
        proxy_weight=weight,
    )
    assert risk.shape == (1, 2, 5, 8)
    assert path.shape == (1, 2, 5, 8, 8)
    assert torch.all(risk[..., 1:] <= risk[..., :-1] + 1e-6)
    assert torch.allclose(risk[..., -1], torch.zeros_like(risk[..., -1]))


def test_robust_density_tiebreak_preserves_primary_near_optimum():
    generator = torch.Generator().manual_seed(11)
    keys = torch.randn(1, 1, 4, 8, 16, generator=generator)
    weight = torch.rand(1, 1, 4, 8, generator=generator) + 0.1
    for epsilon in (0.0, 0.01, 0.1):
        risk, path = fit_agglomerative_self_lse_paths(
            keys,
            temperatures=(0.25,),
            cost_mode="importance_relative",
            proxy_weight=weight,
            robust_tiebreak_epsilon=epsilon,
        )
        assert torch.isfinite(risk).all()
        assert path[..., -1, :].amax().item() == 7
        assert torch.all(risk[..., 1:] <= risk[..., :-1] + 1e-6)
