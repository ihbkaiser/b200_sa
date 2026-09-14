import types

import pytest
import torch

from models.adaptive_centroid_cache import AdaptiveCentroidLSECache


def config():
    return types.SimpleNamespace(
        hidden_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        head_dim=128,
    )


def test_method_forces_zero_outlier_blocks():
    cache = AdaptiveCentroidLSECache(
        config(), max_length=256, sparse_budget=16, chunk_size=8,
        rank=8, n_centroids=8,
        centroid_method="self_lse_adaptive_iso", split_fraction=0.25,
        device="cpu",
    )
    assert cache.outlier_chunk == 0


def test_method_rejects_nonzero_outlier_blocks():
    with pytest.raises(ValueError, match="forbids ShadowKV outlier"):
        AdaptiveCentroidLSECache(
            config(), max_length=256, sparse_budget=16, chunk_size=8,
            rank=8, n_centroids=8,
            centroid_method="self_lse_adaptive_iso", split_fraction=0.25,
            outlier_chunk=1, device="cpu",
        )


def test_prefix4_is_additive_and_excluded_from_dynamic_retrieval():
    cache = AdaptiveCentroidLSECache(
        config(), max_length=256, sparse_budget=16, chunk_size=8,
        rank=8, n_centroids=8,
        centroid_method="self_lse_adaptive_iso", split_fraction=0.25,
        prefix_chunks=4, device="cpu",
    )
    assert cache.prefix_tokens == 32
    assert cache.sparse_budget == 16
    assert cache.selected_chunk_idx.shape[-1] == 2

    cache.k_landmark_idx = torch.arange(8).view(1, 1, 1, 8).expand(
        1, 1, 2, 8
    ).clone()
    cache.select_sets = 2
    scores = torch.tensor(
        [[[100.0, 90.0, 80.0, 70.0, 1.0, 2.0, 3.0, 4.0],
          [100.0, 90.0, 80.0, 70.0, 4.0, 3.0, 2.0, 1.0]]]
    )
    selected = cache._select_router_chunks(0, scores)
    assert torch.equal(selected[0, 0].sort().values, torch.tensor([6, 7]))
    assert torch.equal(selected[0, 1].sort().values, torch.tensor([4, 5]))
    assert torch.all(selected >= 4)

    logits = torch.zeros(1, 2, 2, 1, 8)
    masked = cache._mask_router_block_logits(0, logits)
    assert torch.isneginf(masked[..., :4]).all()
    assert torch.isfinite(masked[..., 4:]).all()


def test_method_gathers_exact_post_rope_keys_without_shadowkv_svd():
    torch.manual_seed(2)
    cache = AdaptiveCentroidLSECache(
        config(), max_length=256, sparse_budget=16, chunk_size=8,
        rank=8, n_centroids=8,
        centroid_method="self_lse_adaptive_iso", split_fraction=0.25,
        prefix_chunks=4, device="cpu", dtype=torch.float32,
    )
    keys = torch.randn(1, 2, 128, 128)
    values = torch.randn_like(keys)
    cache.get_svd(keys, layer_idx=0)
    assert cache.U is None
    assert cache.SV is None

    cache.prefill_kv_cache(values, 0, keys)
    # 12 prompt blocks minus four exact-prefix blocks leaves eight router
    # candidates. At x=0.25 exactly two receive a second component.
    assert torch.all(cache.router_component_count[0][..., 4:].sum(-1) == 10)
    assert torch.all(cache.router_component_count[0][..., :4] == 1)
    positions = torch.tensor(
        [[[40, 41, 42, 43, 44, 45, 46, 47,
           72, 73, 74, 75, 76, 77, 78, 79],
          [48, 49, 50, 51, 52, 53, 54, 55,
           80, 81, 82, 83, 84, 85, 86, 87]]]
    )
    attended = cache.get_key_cache(0, positions)
    expected = keys.gather(
        -2, positions.unsqueeze(-1).expand(-1, -1, -1, 128)
    )
    torch.testing.assert_close(
        attended[:, :, cache.sparse_start:cache.sparse_end], expected
    )
