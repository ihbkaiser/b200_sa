import types

import torch

from models.adaptive_centroid_streaming_cache import (
    StreamingAdaptiveCentroidLSECache,
    _head_trusted_density_weight,
    _macro_angular_kde_maxmed_signal,
    evaluate_angular_radius_path,
    estimate_self_k_block_exposure,
)


def tiny_config():
    return types.SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        head_dim=8,
    )


def test_angular_radius_detects_a_two_direction_block():
    keys = torch.zeros(1, 1, 2, 8, 2)
    keys[0, 0, 0, :, 0] = 1
    keys[0, 0, 1, :4, 0] = 1
    keys[0, 0, 1, 4:, 1] = 1
    path = torch.zeros(1, 1, 2, 8, 8, dtype=torch.uint8)
    for r in range(1, 9):
        # A nested path is sufficient here: at r=2 it isolates the two
        # directions; extra labels only split already coherent members.
        labels = torch.arange(8).clamp_max(r - 1)
        path[..., r - 1, :] = labels
    path[0, 0, 1, 1] = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    risk = evaluate_angular_radius_path(keys, path)
    assert risk.shape == (1, 1, 2, 8)
    torch.testing.assert_close(risk[0, 0, 0], torch.zeros(8), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        risk[0, 0, 1, 0], torch.tensor(1 - 2 ** -0.5), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(risk[0, 0, 1, 1], torch.tensor(0.0))


def test_angular_marginal_allocator_spends_exact_quota(monkeypatch):
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "robust_trimmed")
    monkeypatch.setenv("SHADOWKV_CENTER_ALLOCATION", "angular_marginal")
    torch.manual_seed(2029)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        max_components=8, self_lse_temperatures=(1.0,),
    )
    keys = torch.randn(1, 2, 41, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    counts = cache.component_count[0][..., 1:4]
    assert torch.all(counts.sum(-1) == 4)
    assert torch.all((counts >= 1) & (counts <= 8))


def test_tail_allocator_can_price_a_minimax_placement_path(monkeypatch):
    """Placement and allocation objectives remain independently selectable."""
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "self")
    monkeypatch.setenv("SHADOWKV_CENTER_ALLOCATION", "tail_cvar")
    monkeypatch.setenv("SHADOWKV_SELF_LSE_COST", "max_gap")
    torch.manual_seed(2031)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        max_components=8, self_lse_temperatures=(1.0,),
    )
    keys = torch.randn(1, 2, 41, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    counts = cache.component_count[0][..., 1:4]
    assert torch.all(counts.sum(-1) == 4)
    assert torch.all((counts >= 1) & (counts <= 8))


def test_self_k_exposure_prefers_the_more_addressable_block():
    keys = torch.zeros(1, 1, 2, 8, 2)
    keys[0, 0, 0, :, 0] = 5
    keys[0, 0, 1, :, 1] = 1
    weight = estimate_self_k_block_exposure(
        keys, reference_count=16, temperature=1.0, batch_blocks=1
    )
    assert weight.shape == (1, 1, 2)
    torch.testing.assert_close(weight.mean(-1), torch.ones(1, 1))
    assert weight[0, 0, 0] > weight[0, 0, 1]


def test_exposure_angular_allocator_spends_exact_quota(monkeypatch):
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "robust_trimmed")
    monkeypatch.setenv(
        "SHADOWKV_CENTER_ALLOCATION", "exposure_angular_marginal"
    )
    monkeypatch.setenv("SHADOWKV_EXPOSURE_REFERENCE_COUNT", "16")
    torch.manual_seed(2030)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        max_components=8, self_lse_temperatures=(1.0,),
    )
    keys = torch.randn(1, 2, 41, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    counts = cache.component_count[0][..., 1:4]
    assert torch.all(counts.sum(-1) == 4)
    assert torch.all((counts >= 1) & (counts <= 8))


def test_density_tail_allocator_is_query_free_and_spends_exact_quota(
    monkeypatch,
):
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "robust_trimmed")
    monkeypatch.setenv("SHADOWKV_CENTER_ALLOCATION", "density_tail_cvar")
    torch.manual_seed(2026)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        max_components=8, self_lse_temperatures=(1.0,),
    )
    keys = torch.randn(1, 2, 41, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    counts = cache.component_count[0][..., 1:4]
    assert torch.all(counts.sum(-1) == 4)
    assert torch.all((counts >= 1) & (counts <= 8))

    blocks = keys[..., :40, :].view(1, 2, 5, 8, 8)
    signal = _macro_angular_kde_maxmed_signal(
        blocks, page_blocks=4, reference_count=16, kappa=8.0
    )
    assert signal.shape == (1, 2, 5)
    assert torch.isfinite(signal).all()
    assert torch.all(signal >= 1)

    absolute_signal = _macro_angular_kde_maxmed_signal(
        blocks, page_blocks=4, reference_count=16, kappa=8.0,
        signal_mode="absolute_contrast",
    )
    assert absolute_signal.shape == signal.shape
    assert torch.isfinite(absolute_signal).all()

    absolute_max = _macro_angular_kde_maxmed_signal(
        blocks, page_blocks=4, reference_count=16, kappa=8.0,
        signal_mode="absolute_max",
    )
    assert absolute_max.shape == signal.shape
    assert torch.isfinite(absolute_max).all()
    assert torch.allclose(absolute_max.mean(-1), torch.ones(1, 2))


def test_density_up_only_mode_never_downweights_local_distortion(monkeypatch):
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "robust_trimmed")
    monkeypatch.setenv("SHADOWKV_CENTER_ALLOCATION", "density_tail_cvar")
    monkeypatch.setenv("SHADOWKV_DENSITY_WEIGHT_MODE", "up_only")
    monkeypatch.setenv("SHADOWKV_DENSITY_SHRINKAGE", "1")
    torch.manual_seed(2027)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        max_components=8, self_lse_temperatures=(1.0,),
    )
    keys = torch.randn(1, 2, 41, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    counts = cache.component_count[0][..., 1:4]
    assert torch.all(counts.sum(-1) == 4)
    assert torch.all((counts >= 1) & (counts <= 8))


def test_head_trust_smoothly_falls_back_on_fragile_heads():
    signal = torch.tensor([[[2.0, 0.5], [2.0, 0.5]]])
    # Head zero has ten times the upper-tail 1->2 gain of head one.
    risk = torch.tensor([[[[10.0, 0.0], [10.0, 0.0]],
                          [[1.0, 0.0], [1.0, 0.0]]]])
    weight = _head_trusted_density_weight(
        signal, risk, shrinkage=0.5, power=1.0
    )
    assert torch.allclose(weight[0, 0], torch.tensor([1.05, 0.975]))
    assert torch.allclose(weight[0, 1], torch.tensor([1.5, 0.75]))
    legacy = _head_trusted_density_weight(
        signal, risk, shrinkage=0.5, power=0.0
    )
    assert torch.allclose(legacy, torch.tensor([[[1.5, 0.75], [1.5, 0.75]]]))


def test_density_value_power_validation(monkeypatch):
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "robust_trimmed")
    monkeypatch.setenv("SHADOWKV_CENTER_ALLOCATION", "density_tail_cvar")
    monkeypatch.setenv("SHADOWKV_DENSITY_VALUE_POWER", "-1")
    try:
        StreamingAdaptiveCentroidLSECache(
            tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
            sparse_budget=16, block_size=8, dense_layers=0,
            prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
            max_components=8, self_lse_temperatures=(1.0,),
        )
    except ValueError as error:
        assert "value power" in str(error)
    else:
        raise AssertionError("negative density value power was accepted")


def test_value_aware_density_allocator_spends_exact_quota(monkeypatch):
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "robust_trimmed")
    monkeypatch.setenv("SHADOWKV_CENTER_ALLOCATION", "density_tail_cvar")
    monkeypatch.setenv("SHADOWKV_DENSITY_SIGNAL_MODE", "absolute_max")
    monkeypatch.setenv("SHADOWKV_DENSITY_VALUE_POWER", "1")
    torch.manual_seed(2028)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        max_components=8, self_lse_temperatures=(1.0,),
    )
    keys = torch.randn(1, 2, 41, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    counts = cache.component_count[0][..., 1:4]
    assert torch.all(counts.sum(-1) == 4)
    assert torch.all((counts >= 1) & (counts <= 8))
