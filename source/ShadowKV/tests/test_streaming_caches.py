import math
import types

import pytest
import torch

from models.adaptive_centroid_streaming_cache import (
    StreamingAdaptiveCentroidLSECache,
)
from models.quest_streaming_cache import StreamingQuestCache
from models.retroinfer_streaming_cache import StreamingRetroInferReferenceCache
from models.exact_block_streaming_cache import StreamingExactBlockOracleCache
from models.exact_totalmass_rerank_cache import StreamingExactTotalMassRerankCache
from models.adaptive_centroid_triton import two_slot_block_logits
from models.adaptive_centroid_incremental import IncrementalTwoCentroidGraph
from models.centroid_router_cache import (
    allocate_minimax_centroid_counts,
    fit_lazy_exact_self_lse_allocation,
)
from models.adaptive_centroid_streaming_cache import _padded_adaptive_centroids
from models.offload_gather import gather_blocks_reuse_uva
from models.kv_cache import ShadowKVCache_CPU


def tiny_config():
    return types.SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        head_dim=8,
    )


def offload_config():
    return types.SimpleNamespace(
        hidden_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        head_dim=128,
    )


def test_adaptive_router_short_context_is_exact_during_warmup():
    torch.manual_seed(73)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=64, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.0,
        query_group_mean=True,
    )
    keys = torch.randn(1, 2, 8, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    query = torch.randn(1, 4, 1, 8)
    selected_k, selected_v = cache.select_key_value_cache(0, query)
    torch.testing.assert_close(selected_k, keys)
    torch.testing.assert_close(selected_v, values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shadowkv_cpu_short_context_dense_warmup_is_exact():
    cache = ShadowKVCache_CPU(
        tiny_config(), max_length=64, device="cuda:0", dtype=torch.float32,
        sparse_budget=16, chunk_size=8, rank=8, outlier_chunk=0,
    )
    keys = torch.randn(1, 2, 8, 8, device="cuda")
    values = torch.randn_like(keys)
    for layer in range(2):
        cache.prefill_dense_warmup(values, layer, keys)
    cache.H2D()

    next_key = torch.randn(1, 2, 1, 8, device="cuda")
    next_value = torch.randn_like(next_key)
    for layer in range(2):
        cache.update_kv_cache(
            next_key, next_value, layer, key_states_prerope=next_key
        )
        dense_k, dense_v = cache.get_dense_cache(layer)
        torch.testing.assert_close(dense_k, torch.cat((keys, next_key), -2))
        torch.testing.assert_close(dense_v, torch.cat((values, next_value), -2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_two_slot_triton_router_matches_reference_scores_and_selection():
    torch.manual_seed(314)
    blocks = 4096
    query = torch.randn(
        1, 8, 4, 1, 128, device="cuda", dtype=torch.bfloat16
    )
    centers = torch.randn(
        1, 8, blocks, 2, 128, device="cuda", dtype=torch.bfloat16
    ) * 0.4
    log_counts = torch.zeros(
        1, 8, blocks, 2, device="cuda", dtype=torch.bfloat16
    )
    log_counts[..., 1].masked_fill_(
        torch.rand(1, 8, blocks, device="cuda") > 0.25, float("-inf")
    )
    alpha = torch.rand(1, 8, blocks, 2, device="cuda") * 0.02
    actual_logits = two_slot_block_logits(
        query, centers, log_counts, alpha
    )
    component = torch.einsum(
        "bhgqd,bhnrd->bhgqnr", query, centers
    ).float() / 128**0.5
    component = component + log_counts.float()[:, :, None, None]
    norm2 = query.float().square().sum(-1) / 128
    component = component + norm2[..., None, None] * alpha[:, :, None, None]
    expected_logits = torch.logsumexp(component, -1).squeeze(-2)
    actual_scores = torch.softmax(actual_logits, -1).sum(2)
    expected_scores = torch.softmax(expected_logits, -1).sum(2)
    torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=1e-5)
    actual_top = actual_scores.topk(256, -1).indices
    expected_top = expected_scores.topk(256, -1).indices
    overlap = (
        actual_top[..., :, None] == expected_top[..., None, :]
    ).any(-1).float().mean()
    # Triton's reduction/log implementation is not bitwise identical across
    # SM86 and SM89.  The score tensor is already bounded above; require the
    # selected set to agree except at the numerically tied top-k boundary.
    # On the fixed stress seed SM89 differs by four of 2048 selected blocks
    # (99.8% overlap), whose reference boundary margins are below 6e-7.
    assert overlap.item() >= 0.995


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_incremental_cuda_graph_matches_exact_two_centroid_fit():
    torch.manual_seed(2718)
    keys = torch.randn(
        1, 8, 1, 8, 128, device="cuda", dtype=torch.bfloat16
    )
    threshold = torch.rand(1, 8, device="cuda") * 0.2
    risk, assignments, _ = fit_lazy_exact_self_lse_allocation(
        keys, extra_fraction=1.0, temperatures=(1.0,)
    )
    counts = 1 + (risk[..., 0] >= threshold[..., None]).long()
    expected = _padded_adaptive_centroids(keys, assignments, counts, 2)

    fitter = IncrementalTwoCentroidGraph(
        batch_size=1,
        heads=8,
        dim=128,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    centers, logs, alpha, actual_counts, actual_risk = fitter.run(
        keys, threshold
    )
    torch.cuda.synchronize()
    for actual, reference in zip((centers, logs, alpha), expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_counts, counts, rtol=0, atol=0)
    torch.testing.assert_close(actual_risk, risk[..., 0], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streaming_cpu_offload_matches_gpu_backing():
    torch.manual_seed(101)
    common = dict(
        max_length=64,
        device="cuda:0",
        dtype=torch.bfloat16,
        sparse_budget=8,
        block_size=8,
        dense_layers=0,
        prefix_tokens=8,
        recent_tokens=8,
        extra_fraction=0.25,
    )
    gpu = StreamingAdaptiveCentroidLSECache(
        offload=False, **common, config=offload_config()
    )
    cpu = StreamingAdaptiveCentroidLSECache(
        offload=True,
        offload_backend="uva",
        **common,
        config=offload_config(),
    )
    keys = torch.randn(1, 2, 25, 128, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    gpu.prefill_kv_cache(values, 0, keys)
    cpu.prefill_kv_cache(values, 0, keys)
    cpu.H2D()

    # The prompt tail plus seven generated tokens seals the same streaming
    # block in both paths; metadata is built from the GPU staging block.
    new_keys = torch.randn(1, 2, 7, 128, device="cuda", dtype=torch.bfloat16)
    new_values = torch.randn_like(new_keys)
    gpu.update_kv_cache(new_keys, new_values, 0)
    cpu.update_kv_cache(new_keys, new_values, 0)
    query = torch.randn(1, 4, 1, 128, device="cuda", dtype=torch.bfloat16)
    gpu_ids = gpu.get_retrieval_position_ids(0, query)
    cpu_ids = cpu.get_retrieval_position_ids(0, query)
    torch.testing.assert_close(cpu_ids, gpu_ids)
    gpu_k, gpu_v = gpu.get_key_value_cache(0, gpu_ids)
    cpu_k, cpu_v = cpu.get_key_value_cache(0, cpu_ids)
    torch.cuda.synchronize()
    torch.testing.assert_close(cpu_k, gpu_k)
    torch.testing.assert_close(cpu_v, gpu_v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_uva_block_gather_reuses_overlap_and_fetches_replacements():
    """The temporal gather preserves current top-k order and exact ranges."""
    torch.manual_seed(707)
    batch, heads, length, dim = 1, 2, 64, 128
    block_size, selected_blocks, capacity = 8, 2, 24
    source_k = torch.randn(
        batch, heads, length, dim, dtype=torch.bfloat16, pin_memory=True
    )
    source_v = torch.randn_like(source_k, pin_memory=True)
    key_banks = [
        torch.empty(
            batch, heads, capacity, dim, device="cuda", dtype=torch.bfloat16
        )
        for _ in range(2)
    ]
    value_banks = [torch.empty_like(bank) for bank in key_banks]
    id_banks = [
        torch.full(
            (batch, heads, selected_blocks),
            -1,
            device="cuda",
            dtype=torch.long,
        )
        for _ in range(2)
    ]
    first = torch.tensor(
        [[[1, 3], [2, 4]]], device="cuda", dtype=torch.long
    )
    exact_ranges = ((0, 4), (60, 64))
    first_k, first_v = gather_blocks_reuse_uva(
        source_k,
        source_v,
        key_banks[1],
        value_banks[1],
        id_banks[1],
        first,
        key_banks[0],
        value_banks[0],
        id_banks[0],
        block_size=block_size,
        exact_ranges=exact_ranges,
    )
    torch.cuda.synchronize()

    def expected(source, blocks):
        rows = []
        for head in range(heads):
            positions = []
            for block in blocks[0, head].cpu().tolist():
                positions.extend(range(block * block_size, (block + 1) * block_size))
            positions.extend(range(0, 4))
            positions.extend(range(60, 64))
            rows.append(source[0, head, positions])
        return torch.stack(rows).unsqueeze(0)

    torch.testing.assert_close(first_k.cpu(), expected(source_k, first))
    torch.testing.assert_close(first_v.cpu(), expected(source_v, first))

    # Reorder one old block and introduce one replacement per head.  Corrupt
    # the CPU copy of the shared blocks: the second result must still carry the
    # original values from the preceding GPU bank, proving that reuse is real.
    second = torch.tensor(
        [[[3, 5], [4, 1]]], device="cuda", dtype=torch.long
    )
    original_k = source_k.clone()
    original_v = source_v.clone()
    source_k[0, 0, 24:32].fill_(123)
    source_v[0, 0, 24:32].fill_(123)
    source_k[0, 1, 32:40].fill_(123)
    source_v[0, 1, 32:40].fill_(123)
    second_k, second_v = gather_blocks_reuse_uva(
        source_k,
        source_v,
        key_banks[0],
        value_banks[0],
        id_banks[0],
        second,
        key_banks[1],
        value_banks[1],
        id_banks[1],
        block_size=block_size,
        exact_ranges=exact_ranges,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(second_k.cpu(), expected(original_k, second))
    torch.testing.assert_close(second_v.cpu(), expected(original_v, second))
    torch.testing.assert_close(id_banks[1], second)


def test_streaming_quest_indexes_decode_block_without_double_attention():
    torch.manual_seed(0)
    cache = StreamingQuestCache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=8, page_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8,
    )
    keys = torch.randn(1, 2, 25, 8)
    values = torch.randn_like(keys)
    for layer in range(2):
        cache.prefill_kv_cache(values, layer, keys)

    # The five-token prompt tail and three generated tokens form block 3.
    new_keys = torch.randn(1, 2, 7, 8)
    new_values = torch.randn_like(new_keys)
    query = torch.randn(1, 4, 1, 8)
    for layer in range(2):
        cache.update_kv_cache(new_keys, new_values, layer)
        expected = cache.k_cache[layer, :, :, 24:32]
        torch.testing.assert_close(
            cache.page_min[layer, :, :, 3], expected.amin(dim=-2)
        )
        torch.testing.assert_close(
            cache.page_max[layer, :, :, 3], expected.amax(dim=-2)
        )
        dynamic = cache.get_retrieval_position_ids(layer, query)
        attended = cache.get_key_cache(layer, dynamic)
        # dynamic 8 + exact prefix 8 + exact recent 8
        assert attended.shape[-2] == 24
        assert torch.all(dynamic >= 8)
        assert torch.all(dynamic < 24)

    assert cache.get_kv_len() == 32


def test_streaming_batched_update_preserves_tail_across_two_sealed_blocks():
    torch.manual_seed(9)
    cache = StreamingQuestCache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=8, page_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8,
    )
    keys = torch.randn(1, 2, 25, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    first_keys = torch.randn(1, 2, 10, 8)
    first_values = torch.randn_like(first_keys)
    cache.update_kv_cache(first_keys, first_values, 0)
    assert cache.block_state[0].total_tokens == 35
    second_keys = torch.randn(1, 2, 5, 8)
    second_values = torch.randn_like(second_keys)
    cache.update_kv_cache(second_keys, second_values, 0)
    assert cache.block_state[0].total_tokens == 40
    for block_id in (3, 4):
        expected = cache.k_cache[0, :, :, block_id * 8 : (block_id + 1) * 8]
        torch.testing.assert_close(
            cache.page_min[0, :, :, block_id], expected.amin(dim=-2)
        )
        torch.testing.assert_close(
            cache.page_max[0, :, :, block_id], expected.amax(dim=-2)
        )


def test_streaming_adaptive_router_allocates_and_scores_new_blocks(monkeypatch):
    monkeypatch.setenv("SHADOWKV_DEFER_SEALED_BUILD", "0")
    torch.manual_seed(1)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        self_lse_temperatures=(1.0,),
    )
    assert cache.router_slots == 8
    keys = torch.randn(1, 2, 41, 8)
    values = torch.randn_like(keys)
    for layer in range(2):
        cache.prefill_kv_cache(values, layer, keys)
        # Prefix block0 and recent block4 are exact.  The one extra component
        # is allocated over candidate blocks1:4 only.
        assert torch.all(cache.component_count[layer, :, :, 1:4].sum(-1) == 4)

    new_keys = torch.randn(1, 2, 7, 8)
    new_values = torch.randn_like(new_keys)
    query = torch.randn(1, 4, 1, 8)
    for layer in range(2):
        cache.update_kv_cache(new_keys, new_values, layer)
        assert torch.all(cache.component_count[layer, :, :, 5] >= 1)
        ids = cache.get_retrieval_position_ids(layer, query)
        assert ids.shape == (1, 2, 16)
        assert cache.get_key_cache(layer, ids).shape[-2] == 32

    assert cache.get_kv_len() == 48


def test_streaming_adaptive_router_amortizes_sealed_layers(monkeypatch):
    monkeypatch.setenv("SHADOWKV_DEFER_SEALED_BUILD", "1")
    torch.manual_seed(17)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        self_lse_temperatures=(1.0,),
    )
    keys = torch.randn(1, 2, 41, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 1, keys)

    sealed = torch.randn(1, 2, 7, 8)
    cache.update_kv_cache(sealed, torch.randn_like(sealed), 1)
    assert cache.deferred_block_id[1] == 5
    assert torch.all(cache.component_count[1, :, :, 5] == 0)

    one = torch.randn(1, 2, 1, 8)
    cache.update_kv_cache(one, torch.randn_like(one), 1)
    assert cache.deferred_block_id[1] == 5
    cache.update_kv_cache(one, torch.randn_like(one), 1)
    assert cache.deferred_block_id[1] == -1
    assert torch.all(cache.component_count[1, :, :, 5] >= 1)


def test_streaming_adaptive_router_rejects_obsolete_two_center_cap():
    with pytest.raises(ValueError, match="full 1..block_size"):
        StreamingAdaptiveCentroidLSECache(
            tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
            sparse_budget=16, block_size=8, dense_layers=0,
            prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
            self_lse_temperatures=(1.0,), max_components=2,
        )


def test_streaming_adaptive_router_can_disable_dispersion_correction(
    monkeypatch,
):
    monkeypatch.setenv("SHADOWKV_CENTER_DISPERSION_CORRECTION", "0")
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        self_lse_temperatures=(1.0,),
    )
    keys = torch.randn(1, 2, 41, 8)
    cache.prefill_kv_cache(torch.randn_like(keys), 0, keys)
    assert not cache.center_dispersion_correction
    assert torch.count_nonzero(cache.router_alpha[0]) == 0


def test_query_mean_selector_skips_softmax_without_changing_topk(monkeypatch):
    """A singleton query's softmax and logits induce the same block order."""
    torch.manual_seed(311)
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "self")
    monkeypatch.setenv(
        "SHADOWKV_CENTER_ALLOCATION", "tail_absolute_rate_distortion"
    )
    monkeypatch.setenv("SHADOWKV_CENTER_DISPERSION_CORRECTION", "0")
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, query_group_mean=True,
        self_lse_temperatures=(1.0,), max_components=8,
    )
    keys = torch.randn(1, 2, 73, 8)
    cache.prefill_kv_cache(torch.randn_like(keys), 0, keys)
    query = torch.randn(1, 4, 1, 8)
    first, last = cache.block_state[0].candidate_block_range
    normalized = cache._score_blocks(0, query, first, last)
    expected = normalized.topk(cache.select_blocks, dim=-1).indices + first
    actual = cache._select_block_ids(0, query)
    torch.testing.assert_close(actual, expected)


def test_canonical_router_does_not_materialize_unused_value_means(monkeypatch):
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "self")
    monkeypatch.setenv(
        "SHADOWKV_CENTER_ALLOCATION", "tail_absolute_rate_distortion"
    )
    monkeypatch.delenv("SHADOWKV_STORE_ROUTER_VALUE_MEAN", raising=False)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, query_group_mean=True,
        self_lse_temperatures=(1.0,), max_components=8,
    )
    keys = torch.randn(1, 2, 73, 8)
    values = torch.randn_like(keys) + 3
    cache.prefill_kv_cache(values, 0, keys)
    assert not cache.store_router_value_mean
    assert torch.count_nonzero(cache.router_value_mean[0]) == 0


def test_streaming_adaptive_router_compact_metadata_matches_dense_and_quantizes():
    torch.manual_seed(31)
    kwargs = dict(
        max_length=256, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        self_lse_temperatures=(1.0,),
    )
    dense = StreamingAdaptiveCentroidLSECache(tiny_config(), **kwargs)
    compact = StreamingAdaptiveCentroidLSECache(
        tiny_config(), compact_metadata=True, center_bits=16, **kwargs
    )
    int8 = StreamingAdaptiveCentroidLSECache(
        tiny_config(), compact_metadata=True, center_bits=8, **kwargs
    )
    int4 = StreamingAdaptiveCentroidLSECache(
        tiny_config(), compact_metadata=True, center_bits=4, **kwargs
    )
    keys = torch.randn(1, 2, 57, 8)
    values = torch.randn_like(keys)
    query = torch.randn(1, 4, 1, 8)
    for cache in (dense, compact, int8, int4):
        cache.prefill_kv_cache(values, 0, keys)

    first, last = dense.block_state[0].candidate_block_range
    dense_score = dense._score_blocks(0, query, first, last)
    compact_score = compact._score_blocks(0, query, first, last)
    int8_score = int8._score_blocks(0, query, first, last)
    int4_score = int4._score_blocks(0, query, first, last)
    torch.testing.assert_close(compact_score, dense_score, rtol=1e-6, atol=1e-7)
    assert torch.max(torch.abs(int8_score - dense_score)) < 2e-3
    assert torch.max(torch.abs(int4_score - dense_score)) < 3e-2

    expected = int(compact.component_count[0].sum(-1).max().item())
    assert compact.router_component_used[0] == expected
    assert compact.router_centroids[0].shape[-1] == 8
    assert int8.router_centroids[0].dtype == torch.int8
    assert int4.router_centroids[0].dtype == torch.uint8
    assert int4.router_centroids[0].shape[-1] == 4


def test_rank16_int4_refinement_uses_codes_after_coarse_block_shortlist(
    monkeypatch,
):
    torch.manual_seed(32)
    monkeypatch.setenv("STREAMING_REFINE_SKETCH_RANK", "4")
    monkeypatch.setenv("STREAMING_REFINE_SKETCH_BITS", "4")
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=256, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        self_lse_temperatures=(1.0,), compact_metadata=True,
        center_bits=4, query_group_mean=True, max_components=8,
        refine_tokens=True, refine_factor=2,
    )
    keys = torch.randn(1, 2, 57, 8)
    values = torch.randn_like(keys)
    prompt_query = torch.randn(1, 4, 19, 8)
    router_query = cache.prepare_prefill_query(prompt_query)
    cache.prefill_kv_cache(values, 0, keys, router_query)

    assert cache.refine_key_codes[0].dtype == torch.uint8
    assert cache.refine_key_codes[0].shape[-1] == 2
    assert cache.refine_key_scale[0].shape[-1] == 8
    assert cache._refine_prompt_query[0] is None
    query = torch.randn(1, 4, 1, 8)
    position = cache.get_retrieval_position_ids(0, query)
    assert position.shape == (1, 2, 16)
    assert torch.all(position >= 8)


def test_key_pca_refinement_basis_is_orthonormal(monkeypatch):
    torch.manual_seed(33)
    monkeypatch.setenv("STREAMING_REFINE_SKETCH_RANK", "4")
    monkeypatch.setenv("STREAMING_REFINE_SKETCH_BITS", "4")
    monkeypatch.setenv("STREAMING_REFINE_SKETCH_BASIS", "key_pca")
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=256, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        self_lse_temperatures=(1.0,), compact_metadata=True,
        center_bits=8, query_group_mean=True, max_components=8,
        refine_tokens=True, refine_factor=2,
    )
    keys = torch.randn(1, 2, 57, 8)
    values = torch.randn_like(keys)
    prompt_query = torch.randn(1, 4, 19, 8)
    router_query = cache.prepare_prefill_query(prompt_query)
    cache.prefill_kv_cache(values, 0, keys, router_query)

    query_factor = cache.refine_query_factor[0]
    key_factor = cache.refine_key_factor[0]
    assert torch.allclose(query_factor, key_factor)
    gram = key_factor.transpose(-1, -2) @ key_factor
    eye = torch.eye(4).expand_as(gram)
    assert torch.allclose(gram, eye, atol=1e-5, rtol=1e-5)


def test_streaming_adaptive_router_qone_uses_compact_variable_order(monkeypatch):
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "qone")
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=256, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        self_lse_temperatures=(1.0,), compact_metadata=True,
        max_components=8, center_bits=16,
    )
    keys = torch.randn(1, 2, 57, 8)
    values = torch.randn_like(keys)
    query = torch.randn(1, 4, 1, 8)
    cache.prefill_kv_cache(values, 0, keys, query)

    first, last = cache.block_state[0].candidate_block_range
    active = cache.component_count[0, :, :, : cache.block_state[0].sealed_blocks]
    assert torch.all(active >= 1)
    assert int(active.max()) <= 8
    assert int(active[..., first:last].sum(-1)[0, 0]) == (
        (last - first) + round(0.25 * (last - first))
    )
    score = cache._score_blocks(0, query, first, last)
    assert score.shape == (1, 2, last - first)
    assert torch.isfinite(score).all()


@pytest.mark.parametrize("center_bits,atol", [(16, 1e-4), (8, 1e-4), (4, 1e-4)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compact_triton_router_matches_dequantized_reference(center_bits, atol):
    torch.manual_seed(37)
    cache = StreamingAdaptiveCentroidLSECache(
        offload_config(), max_length=2048, device="cuda", dtype=torch.bfloat16,
        sparse_budget=256, block_size=8, dense_layers=0,
        prefix_tokens=32, recent_tokens=32, extra_fraction=0.25,
        self_lse_temperatures=(1.0,), compact_metadata=True,
        center_bits=center_bits, router_backend="triton",
    )
    keys = torch.randn(1, 2, 1031, 128, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    query = torch.randn(1, 4, 1, 128, device="cuda", dtype=torch.bfloat16)
    first, last = cache.block_state[0].candidate_block_range

    cache.router_backend = "torch"
    expected = cache._score_blocks(0, query, first, last)
    cache.router_backend = "triton"
    actual = cache._score_blocks(0, query, first, last)
    torch.testing.assert_close(actual, expected, rtol=0, atol=atol)
    actual_top = actual.topk(32, -1).indices
    expected_top = expected.topk(32, -1).indices
    overlap = (
        actual_top[..., :, None] == expected_top[..., None, :]
    ).any(-1).float().mean()
    # Random inputs make the top-k boundary deliberately dense; the largest
    # probability error is already bounded above and selection may differ only
    # for those near-ties.
    assert overlap.item() >= 0.98


def test_streaming_adaptive_router_supports_four_components_on_average():
    torch.manual_seed(2)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=3.0,
        self_lse_temperatures=(1.0,),
    )
    keys = torch.randn(1, 2, 41, 8)
    values = torch.randn_like(keys)
    for layer in range(2):
        cache.prefill_kv_cache(values, layer, keys)
        # Candidate blocks 1:4 receive nine upgrades in total: twelve
        # components across three blocks, while exact regions do not spend it.
        counts = cache.component_count[layer, :, :, 1:4]
        assert torch.all(counts.sum(-1) == 12)
        assert torch.all((counts >= 1) & (counts <= 8))

    # A newly sealed block is provisionally assigned using the same global
    # upgrade threshold, ready for when it leaves the rolling recent window.
    new_keys = torch.randn(1, 2, 7, 8)
    new_values = torch.randn_like(new_keys)
    for layer in range(2):
        cache.update_kv_cache(new_keys, new_values, layer)
        count = cache.component_count[layer, :, :, 5]
        assert torch.all((count >= 1) & (count <= 8))


def test_streaming_adaptive_router_keeps_mean_two_boundary_multilevel():
    torch.manual_seed(2)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=1.0,
        self_lse_temperatures=(1.0,),
    )
    assert cache.router_slots == 8
    keys = torch.randn(1, 2, 41, 8)
    values = torch.randn_like(keys)
    for layer in range(2):
        cache.prefill_kv_cache(values, layer, keys)
        # Candidate blocks 1:4 have a base component each plus three adaptive
        # upgrades: exactly six components, without requiring a uniform 2/2/2
        # allocation.
        counts = cache.component_count[layer, :, :, 1:4]
        assert torch.all(counts.sum(-1) == 6)
        assert torch.all((counts >= 1) & (counts <= 8))


@pytest.mark.parametrize(
    "allocation",
    (
        "residual_first", "residual_gate", "residual_weighted",
        "residual_marginal", "residual_rank_marginal",
    ),
)
def test_residual_allocators_spend_exact_quota_and_allow_multilevel(
    monkeypatch, allocation,
):
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "residual_path")
    monkeypatch.setenv("SHADOWKV_CENTER_ALLOCATION", allocation)
    torch.manual_seed(23)
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
    # Three routed blocks receive exactly round(.25 * 3) == one upgrade.
    assert torch.all(counts.sum(-1) == 4)
    assert torch.all((counts >= 1) & (counts <= 8))


def test_streaming_adaptive_router_can_average_gqa_queries_before_scoring():
    torch.manual_seed(22)
    kwargs = dict(
        max_length=128, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=0, recent_tokens=0, extra_fraction=0.25,
        self_lse_temperatures=(1.0,),
    )
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), query_group_mean=True, **kwargs
    )
    keys = torch.randn(1, 2, 24, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    query = torch.randn(1, 4, 1, 8)

    actual = cache._score_blocks(0, query, 0, 3)
    averaged = query.view(1, 2, 2, 1, 8).mean(dim=(2, 3), keepdim=True)
    centers = cache.router_centroids[0, :, :, :3]
    logits = torch.einsum("bhgqd,bhcrd->bhgqcr", averaged, centers)
    logits = logits.float() / 8**0.5
    logits = logits + cache.router_log_counts[0, :, :, :3].float()[:, :, None, None]
    norm2 = averaged.float().square().sum(-1) / 8
    logits = logits + norm2[..., None, None] * cache.router_alpha[
        0, :, :, :3
    ][:, :, None, None]
    expected = torch.softmax(torch.logsumexp(logits, -1), -1).sum(-2).squeeze(2)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 2, 3)


def test_independent_coverage_allocator_does_not_conserve_layer_budget(
    monkeypatch,
):
    monkeypatch.setenv("SHADOWKV_HEAD_ALLOC_TAU", "1")
    monkeypatch.setenv("SHADOWKV_HEAD_ALLOC_MODE", "coverage")
    monkeypatch.setenv("SHADOWKV_HEAD_ALLOC_COVERAGE", "0.9")
    monkeypatch.setenv("SHADOWKV_HEAD_ALLOC_MIN_BUDGET", "0")
    monkeypatch.setenv("SHADOWKV_HEAD_ALLOC_MAX_BUDGET", "32")
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=64, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=0, recent_tokens=0, extra_fraction=0.25,
        self_lse_temperatures=(1.0,),
    )
    scores = torch.tensor([[
        [0.70, 0.20, 0.10, 0.00],
        [0.30, 0.30, 0.20, 0.20],
    ]])
    counts = cache._allocate_head_blocks(0, scores)
    assert counts.tolist() == [[2, 4]]
    # Uniform B=16 would spend four blocks across the two heads.  Independent
    # stopping spends six because the diffuse head decides for itself.
    assert counts.sum().item() == 6

    # If exact prefix/recent already carry 95% of a head's mass, that head
    # needs no dynamic block when the floor is zero.
    counts = cache._allocate_head_blocks(
        0, scores, candidate_fraction=torch.tensor([[0.5, 0.05]])
    )
    assert counts.tolist() == [[2, 0]]


def test_worst_gqa_coverage_protects_each_sibling(monkeypatch):
    monkeypatch.setenv("SHADOWKV_HEAD_ALLOC_TAU", "1")
    monkeypatch.setenv("SHADOWKV_HEAD_ALLOC_MODE", "coverage_worst")
    monkeypatch.setenv("SHADOWKV_HEAD_ALLOC_COVERAGE", "0.9")
    monkeypatch.setenv("SHADOWKV_HEAD_ALLOC_MIN_BUDGET", "0")
    monkeypatch.setenv("SHADOWKV_HEAD_ALLOC_MAX_BUDGET", "32")
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=64, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=0, recent_tokens=0, extra_fraction=0.25,
        self_lse_temperatures=(1.0,),
    )
    probability = torch.tensor([
        [0.90, 0.10, 1e-8, 1e-8],
        [1e-8, 1e-8, 0.60, 0.40],
    ])
    logits = probability.log()[None, None, :, None].expand(1, 2, 2, 1, 4)
    cache._last_ragged_block_logits = logits
    cache._last_ragged_candidate_fraction_groups = torch.ones(1, 2, 2, 1)
    scores = torch.softmax(logits, -1).amax(dim=(2, 3))
    positions, lengths = cache._ragged_position_ids(0, scores, 0)
    assert lengths.tolist() == [[24, 24]]
    assert positions.shape == (1, 2, 24)
    assert cache._head_allocation_history[-1].tolist() == [[24, 24]]


def test_streaming_adaptive_router_exact_lse_candidate_refinement():
    torch.manual_seed(29)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=64, device="cpu", dtype=torch.float32,
        sparse_budget=8, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        self_lse_temperatures=(1.0,), refine_factor=2.0,
    )
    keys = torch.randn(1, 2, 48, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    query_states = torch.randn(1, 4, 1, 8)

    first, last = cache.block_state[0].candidate_block_range
    query, coarse_logits = cache._block_logits(
        0, query_states, first, last
    )
    coarse_score = cache._reduce_block_logits(coarse_logits)
    candidate_relative = coarse_score.topk(2, dim=-1).indices
    candidate_blocks = candidate_relative + first
    candidate_keys = cache._gather_candidate_keys(0, candidate_blocks)
    exact = torch.logsumexp(
        torch.einsum(
            "bhgqd,bhcsd->bhgqcs", query, candidate_keys
        ) / 8**0.5,
        dim=-1,
    )
    hybrid = coarse_logits.clone()
    scatter = candidate_relative[:, :, None, None].expand(1, 2, 2, 1, 2)
    hybrid.scatter_(-1, scatter, exact)
    candidate_score = torch.softmax(hybrid, -1).mean((2, 3)).gather(
        -1, candidate_relative
    )
    expected = candidate_blocks.gather(
        -1, candidate_score.topk(1, dim=-1).indices
    )

    actual = cache._select_block_ids(0, query_states)
    torch.testing.assert_close(actual, expected)


def test_streaming_adaptive_router_ratio_refinement_matches_block_traffic():
    torch.manual_seed(41)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=96, device="cpu", dtype=torch.float32,
        sparse_budget=8, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        self_lse_temperatures=(1.0,), refine_candidate_ratio=0.75,
    )
    keys = torch.randn(1, 2, 64, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    query_states = torch.randn(1, 4, 1, 8)
    first, last = cache.block_state[0].candidate_block_range
    available = last - first
    seen = []
    original = cache._gather_candidate_keys

    def record(layer_idx, block_ids):
        seen.append(block_ids.shape[-1])
        return original(layer_idx, block_ids)

    cache._gather_candidate_keys = record
    cache._select_block_ids(0, query_states)
    assert seen == [math.ceil(0.75 * available)]


def test_streaming_adaptive_router_can_keep_individual_refined_tokens():
    torch.manual_seed(43)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=96, device="cpu", dtype=torch.float32,
        sparse_budget=8, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        self_lse_temperatures=(1.0,), refine_candidate_ratio=0.75,
        refine_tokens=True,
    )
    keys = torch.randn(1, 2, 64, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    query_states = torch.randn(1, 4, 1, 8)
    ids = cache.get_retrieval_position_ids(0, query_states)
    assert ids.shape == (1, 2, 8)
    assert cache.block_selection is False
    assert torch.unique(ids[0, 0]).numel() == 8
    first, last = cache.block_state[0].candidate_block_range
    assert torch.all(ids >= first * cache.block_size)
    assert torch.all(ids < last * cache.block_size)


def test_streaming_adaptive_router_can_shortlist_exact_components(monkeypatch):
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "angular_path")
    monkeypatch.setenv("STREAMING_REFINE_COMPONENT_CANDIDATES", "1")
    torch.manual_seed(44)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=96, device="cpu", dtype=torch.float32,
        sparse_budget=8, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=7.0,
        self_lse_temperatures=(1.0,), refine_factor=2.0,
        refine_tokens=True, max_components=8, query_group_mean=True,
    )
    keys = torch.randn(1, 2, 64, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    query_states = torch.randn(1, 4, 1, 8)
    ids = cache.get_retrieval_position_ids(0, query_states)
    assert ids.shape == (1, 2, 8)
    assert torch.unique(ids[0, 0]).numel() == 8
    first, last = cache.block_state[0].candidate_block_range
    assert torch.all(ids >= first * cache.block_size)
    assert torch.all(ids < last * cache.block_size)
    mean_query = query_states.view(1, 2, 2, 1, 8).mean(dim=(2, 3))
    candidate_keys = keys[:, :, first * 8:last * 8]
    exact_score = torch.einsum(
        "bhd,bhnd->bhn", mean_query, candidate_keys
    ) / math.sqrt(8)
    expected = first * 8 + exact_score.topk(8, dim=-1).indices
    torch.testing.assert_close(
        ids.sort(dim=-1).values, expected.sort(dim=-1).values
    )


def test_streaming_adaptive_router_low_mean_budget_can_use_eight_components():
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=256, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, extra_fraction=0.25,
        self_lse_temperatures=(1.0,), max_components=8,
    )
    assert cache.router_slots == 8
    risk = torch.tensor([
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 0.0],
        *([[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.0]] * 27),
    ])
    counts = allocate_minimax_centroid_counts(risk, extra_fraction=0.25)
    assert counts.sum().item() == 35  # 28 base + 7 extra centers
    assert counts[0].item() == 8


def test_querymean_streaming_default_is_clean_fixed_price_rd(monkeypatch):
    for name in (
        "SHADOWKV_SELF_LSE_COST",
        "SHADOWKV_CENTER_ALLOCATION",
        "SHADOWKV_ABSOLUTE_RD_PENALTY",
        "SHADOWKV_TAIL_GAP_CORRECTION_SCALE",
    ):
        monkeypatch.delenv(name, raising=False)
    cache = StreamingAdaptiveCentroidLSECache(
        tiny_config(), max_length=256, device="cpu", dtype=torch.float32,
        sparse_budget=16, block_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8, query_group_mean=True,
    )
    assert cache.extra_fraction == 0.0
    assert cache.self_lse_cost == "mean_gap"
    assert cache.center_allocation == "tail_absolute_rate_distortion"
    assert cache.absolute_rd_penalty == 1.5
    assert cache.tail_gap_correction_scale == 0.0


def test_streaming_retroinfer_update_segment_crosses_prompt_decode_boundary():
    torch.manual_seed(4)
    cache = StreamingRetroInferReferenceCache(
        tiny_config(), max_length=64, device="cpu", dtype=torch.float32,
        sparse_budget=16, prefix_tokens=2, recent_tokens=2,
        update_segment=8, average_cluster_size=2,
        estimation_ratio=0.25, kmeans_iters=2,
    )
    keys = torch.randn(1, 2, 15, 8)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    assert cache.indexed_end[0] == 10
    assert cache.cluster_count[0] == 4

    # Five prompt-buffer tokens plus the first three of these decode tokens
    # create the next 8-token update segment; the final two remain exact recent.
    new_keys = torch.randn(1, 2, 5, 8)
    new_values = torch.randn_like(new_keys)
    cache.update_kv_cache(new_keys, new_values, 0)
    assert cache.indexed_end[0] == 18
    assert cache.cluster_count[0] == 8
    assert torch.all(cache.assignments[0, :, :, 2:18] >= 0)
    assert torch.all(cache.assignments[0, :, :, 18:20] == -1)

    query = torch.randn(1, 4, 1, 8)
    actual = cache.decode_attend(0, query)
    # Expand KV heads to their two GQA heads for a dense reference.
    dense_k = cache.k_cache[0, :, :, :20].repeat_interleave(2, dim=1)
    dense_v = cache.v_cache[0, :, :, :20].repeat_interleave(2, dim=1)
    dense_logits = torch.einsum("bhqd,bhnd->bhqn", query, dense_k) / 8**0.5
    expected = (torch.softmax(dense_logits, dim=-1) @ dense_v).transpose(1, 2)
    # sparse_budget=16 retrieves all eight 2-token clusters, so the result is
    # exactly dense even though the code uses the RetroInfer merge path.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_exact_block_oracles_match_direct_max_and_logsumexp():
    keys = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]]]
    )
    values = torch.zeros_like(keys)
    query = torch.tensor([[[[1.0, 0.5]]]])
    config = types.SimpleNamespace(
        hidden_size=2, num_attention_heads=1, num_key_value_heads=1,
        num_hidden_layers=1, head_dim=2,
    )
    for statistic in ("max", "logsumexp"):
        for normalize_blocks in (False, True):
            cache = StreamingExactBlockOracleCache(
                config, statistic=statistic,
                normalize_blocks=normalize_blocks,
                max_length=8, device="cpu", dtype=torch.float32,
                sparse_budget=2, block_size=2, prefix_tokens=0,
                recent_tokens=0, group_reduce="max",
            )
            cache.prefill_kv_cache(values, 0, keys)
            score = cache._score_blocks(0, query, 0, 2)
            logits = torch.einsum("bhqd,bhnd->bhqn", query, keys) / 2**0.5
            blocks = logits.reshape(1, 1, 1, 2, 2)
            expected = (
                blocks.amax(-1)
                if statistic == "max"
                else torch.logsumexp(blocks, dim=-1)
            ).squeeze(2)
            if normalize_blocks:
                expected = torch.softmax(expected, dim=-1)
            torch.testing.assert_close(score, expected)


def test_exact_block_query_mean_uses_shared_chunked_scorer(monkeypatch):
    monkeypatch.setenv("SHADOWKV_EXACT_QUERY_GROUP_MEAN", "1")
    config = types.SimpleNamespace(
        hidden_size=4, num_attention_heads=2, num_key_value_heads=1,
        num_hidden_layers=1, head_dim=2,
    )
    keys = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]]]
    )
    values = torch.zeros_like(keys)
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    cache = StreamingExactTotalMassRerankCache(
        config, statistic="logsumexp", normalize_blocks=True,
        max_length=8, device="cpu", dtype=torch.float32,
        sparse_budget=2, block_size=2, prefix_tokens=0,
        recent_tokens=0, group_reduce="max",
    )
    cache.prefill_kv_cache(values, 0, keys)
    actual = cache._score_blocks(0, query, 0, 2)
    mean_query = query.view(1, 1, 2, 1, 2).mean((2, 3), keepdim=True)
    logits = torch.einsum(
        "bhgqd,bhnd->bhgqn", mean_query, keys
    ) / 2**0.5
    expected = torch.softmax(
        torch.logsumexp(logits.view(1, 1, 1, 1, 2, 2), dim=-1),
        dim=-1,
    ).sum(-2).squeeze(2)
    torch.testing.assert_close(actual, expected)


def test_gather_buffer_holds_the_widest_exact_region():
    """The exact region grows for a whole interval before a flush shrinks it.

    Sizing the gather buffer for one block was correct only when the flush
    interval *was* the block size.  At a wider interval the region grows by
    one token per generated token, and a buffer sized for the local window
    alone overruns partway through the first generation.
    """
    block, budget, prefix, local, interval = 8, 64, 8, 32, 32
    cache = StreamingQuestCache(
        tiny_config(), max_length=1024, device="cpu", dtype=torch.float32,
        sparse_budget=budget, page_size=block, dense_layers=0,
        prefix_tokens=prefix, recent_tokens=local, update_interval=interval,
    )
    keys = torch.randn(1, 2, 256, 8)
    cache.prefill_kv_cache(torch.randn_like(keys), 0, keys)
    state = cache.block_state[0]
    step = torch.randn(1, 2, 1, 8)
    widest = 0
    for _ in range(4 * interval):
        cache.update_kv_cache(step, torch.randn_like(step), 0)
        selected = min(state.active_blocks, budget // block) * block
        widest = max(widest, selected + state.exact_tokens)
    assert widest > budget + prefix + local + block, (
        "the frame under test must actually exercise the buffer region"
    )
    assert widest <= cache.gather_capacity


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_id_fallback_gathers_the_same_rows_as_the_reuse_kernel():
    """A kernel failure must degrade to the same rows, not to plausible junk.

    ``select_key_value_cache`` passes block ids when the reuse kernel is
    eligible.  The portable fallback reads token positions, so without an
    expansion it would gather token ``b`` for every block ``b`` and return a
    cache that looks healthy and is wrong.
    """
    import models.streaming_cache as streaming_cache

    torch.manual_seed(31)
    cache = StreamingQuestCache(
        offload_config(), max_length=256, device="cuda:0",
        dtype=torch.bfloat16, sparse_budget=32, page_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=16, update_interval=16,
        offload=True, offload_backend="auto",
    )
    keys = torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.bfloat16)
    cache.prefill_kv_cache(torch.randn_like(keys), 0, keys)
    query = torch.randn(1, 4, 1, 128, device="cuda", dtype=torch.bfloat16)

    kernel_k, kernel_v = cache.select_key_value_cache(0, query)
    kernel_k, kernel_v = kernel_k.clone(), kernel_v.clone()

    def refuse(*args, **kwargs):
        raise RuntimeError("no compiler on this machine")

    original = streaming_cache.gather_blocks_reuse_uva
    streaming_cache.gather_blocks_reuse_uva = refuse
    try:
        fallback_k, fallback_v = cache.select_key_value_cache(0, query)
    finally:
        streaming_cache.gather_blocks_reuse_uva = original

    torch.testing.assert_close(fallback_k, kernel_k)
    torch.testing.assert_close(fallback_v, kernel_v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cross_step_reuse_changes_traffic_but_not_rows(monkeypatch):
    """Reuse is a transport optimization: the gathered rows must not move.

    A runtime table may switch it off to compare methods at equal transport,
    so switching it off must cost nothing but time.
    """
    torch.manual_seed(17)

    def build(enabled):
        monkeypatch.setenv("STREAMING_GATHER_REUSE", "1" if enabled else "0")
        cache = StreamingQuestCache(
            offload_config(), max_length=256, device="cuda:0",
            dtype=torch.bfloat16, sparse_budget=32, page_size=8,
            dense_layers=0, prefix_tokens=8, recent_tokens=16,
            update_interval=16, offload=True, offload_backend="auto",
        )
        assert cache.gather_reuse is enabled
        return cache

    keys = torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    step_k = torch.randn(1, 2, 1, 128, device="cuda", dtype=torch.bfloat16)
    step_v = torch.randn_like(step_k)
    query = torch.randn(1, 4, 1, 128, device="cuda", dtype=torch.bfloat16)

    outputs = []
    for enabled in (True, False):
        cache = build(enabled)
        cache.prefill_kv_cache(values, 0, keys)
        rows = []
        for _ in range(6):
            selected_k, selected_v = cache.select_key_value_cache(0, query)
            rows.append((selected_k.clone(), selected_v.clone()))
            cache.update_kv_cache(step_k, step_v, 0)
        outputs.append(rows)

    for (reuse_k, reuse_v), (cold_k, cold_v) in zip(*outputs):
        torch.testing.assert_close(reuse_k, cold_k)
        torch.testing.assert_close(reuse_v, cold_v)


def test_deferred_block_build_selects_exactly_what_immediate_build_selects():
    """Queuing a sealed block until the flush must be invisible to selection.

    A block sealed while the buffer is filling is still inside the exact
    suffix, so it cannot be retrieved before the flush that builds it.  This
    pins that argument: the deferred path and an immediate-build path choose
    the same blocks and return the same rows at every step.
    """
    torch.manual_seed(91)
    block, local, interval, budget = 8, 16, 32, 32

    def build(immediate):
        cache = StreamingQuestCache(
            tiny_config(), max_length=512, device="cpu", dtype=torch.float32,
            sparse_budget=budget, page_size=block, dense_layers=0,
            prefix_tokens=8, recent_tokens=local, update_interval=interval,
        )
        if immediate:
            cache._build_or_defer_blocks = (
                lambda layer, ids, keys: cache._build_blocks(layer, ids, keys)
            )
        return cache

    keys = torch.randn(1, 2, 200, 8)
    values = torch.randn_like(keys)
    steps = [
        (torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8)) for _ in range(80)
    ]
    query = torch.randn(1, 4, 1, 8)

    outputs = []
    for immediate in (False, True):
        cache = build(immediate)
        cache.prefill_kv_cache(values, 0, keys)
        rows = []
        for step_k, step_v in steps:
            cache.update_kv_cache(step_k, step_v, 0)
            selected_k, selected_v = cache.select_key_value_cache(0, query)
            rows.append((selected_k.clone(), selected_v.clone()))
        outputs.append(rows)

    assert len(outputs[0]) == len(outputs[1])
    for step, ((lazy_k, lazy_v), (eager_k, eager_v)) in enumerate(
        zip(*outputs)
    ):
        assert lazy_k.shape == eager_k.shape, f"step {step} width differs"
        torch.testing.assert_close(lazy_k, eager_k)
        torch.testing.assert_close(lazy_v, eager_v)
