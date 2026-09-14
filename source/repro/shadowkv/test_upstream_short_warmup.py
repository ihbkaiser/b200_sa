from unittest.mock import Mock

import torch

from models.magicpig_author_cache import MagicPIGAuthorCache
from models.pqcache_author_cache import PQCacheAuthorCache


def test_pqcache_short_prompt_skips_undefined_kmeans() -> None:
    cache = object.__new__(PQCacheAuthorCache)
    cache.num_layers = 1
    cache.num_key_value_heads = 2
    cache.device = torch.device("cpu")
    cache.sink_tokens = 4
    cache.retrieved_tokens = 8
    cache.recent_tokens = 4
    cache.k_cache = torch.empty(1, 1, 2, 20, 4)
    cache.v_cache = torch.empty_like(cache.k_cache)
    cache.layer_lengths = [0]
    cache.dense_warmup = [False]
    cache.kv_offset = 0
    cache._fit_layer = Mock()

    keys = torch.randn(1, 2, 10, 4)
    values = torch.randn_like(keys)
    cache.prefill_layer(0, keys, values)

    assert cache.dense_warmup == [True]
    assert cache.layer_lengths == [10]
    assert cache.kv_offset == 10
    cache._fit_layer.assert_not_called()
    positions = cache._selected_positions(0, torch.randn(1, 2, 1, 4))
    assert positions.shape == (1, 2, 10)
    assert positions[0, 0].tolist() == list(range(10))


def test_magicpig_short_prompt_does_not_allocate_negative_remote_buffer() -> None:
    cache = object.__new__(MagicPIGAuthorCache)
    cache.num_layers = 2
    cache.sink_tokens = 32
    cache.local_tokens = 32
    cache.kv_offset = 0
    cache._allocated_length = 0
    cache._dense_warmup = False
    cache._dense_keys = [None, None]
    cache._dense_values = [None, None]
    cache.server = Mock()

    keys = torch.randn(1, 2, 48, 4)
    values = torch.randn_like(keys)
    cache.prefill_layer(0, keys, values)
    cache.prefill_layer(1, keys, values)
    cache.begin_decode_step()

    assert cache._dense_warmup
    assert cache.kv_offset == 48
    cache.server.alloc_buffer.assert_not_called()
    cache.server.plan.assert_not_called()
