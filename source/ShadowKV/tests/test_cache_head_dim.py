"""
Every cache class must take head_dim from the config when the config states it.

Qwen3-4B has hidden_size 2560 with 32 heads but head_dim 128, so the
hidden_size // num_heads shortcut yields 80 and the cache silently allocates
tensors with the wrong last dimension. That failure surfaces deep inside a
copy_ during decode, not at construction, so the shapes get their own test.
"""

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.kv_cache import KV_Cache, ShadowKVCache
from models.quest_streaming_cache import StreamingQuestCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# hidden_size // num_attention_heads == 80, but head_dim is 128 -- Qwen3's shape
QWEN3_LIKE = dict(hidden_size=2560, num_attention_heads=32, num_key_value_heads=8,
                  num_hidden_layers=2, head_dim=128)
# no explicit head_dim: 3072 // 24 == 128 -- Llama-3.2's shape
LLAMA_LIKE = dict(hidden_size=3072, num_attention_heads=24, num_key_value_heads=8,
                  num_hidden_layers=2)

CASES = [pytest.param(QWEN3_LIKE, id="qwen3"), pytest.param(LLAMA_LIKE, id="llama32")]


@CUDA
@pytest.mark.parametrize("fields", CASES)
def test_kv_cache_head_dim(fields):
    cache = KV_Cache(types.SimpleNamespace(**fields), max_length=1024)
    assert cache.k_cache.shape[-1] == 128
    assert cache.v_cache.shape[-1] == 128


@CUDA
@pytest.mark.parametrize("fields", CASES)
def test_shadowkv_cache_head_dim(fields):
    cache = ShadowKVCache(types.SimpleNamespace(**fields), max_length=1024,
                          sparse_budget=128, chunk_size=8, rank=16)
    assert cache.head_dim == 128
    assert cache.k_cache_buffer.shape[-1] == 128
    assert cache.v_cache_buffer.shape[-1] == 128


@CUDA
@pytest.mark.parametrize("fields", CASES)
def test_streaming_quest_cache_head_dim(fields):
    cache = StreamingQuestCache(
        types.SimpleNamespace(**fields), max_length=1024, sparse_budget=128
    )
    assert cache.head_dim == 128
    assert cache.k_cache.shape[-1] == 128
