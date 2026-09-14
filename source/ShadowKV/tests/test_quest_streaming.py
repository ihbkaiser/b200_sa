"""Quest's page bound and the streaming-only runtime contract."""

import types

import pytest
import torch

from models.base import LLM
from models.quest_streaming_cache import StreamingQuestCache
from models.quest_triton import quest_page_scores


HEADS, KV_HEADS, LAYERS, HEAD_DIM = 8, 4, 2, 64


def tiny_config():
    return types.SimpleNamespace(
        hidden_size=HEADS * HEAD_DIM,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_hidden_layers=LAYERS,
        head_dim=HEAD_DIM,
    )


def bare_llm(attn_mode):
    llm = LLM.__new__(LLM)
    llm.attn_mode = attn_mode
    llm.max_length = 1024
    llm.device = "cpu"
    llm.dtype = torch.float32
    llm.batch_size = 1
    return llm


def test_only_streaming_quest_is_registered():
    llm = bare_llm("quest_streaming")
    llm.init_kv_cache(
        sparse_budget=128, rank=16, chunk_size=8, config=tiny_config(),
        dense_layers=0, quest_recent_tokens=0,
    )
    assert isinstance(llm.kv_cache, StreamingQuestCache)

    removed = bare_llm("quest")
    with pytest.raises(ValueError, match="Invalid attention mode quest"):
        removed.init_kv_cache(
            sparse_budget=128, rank=16, chunk_size=8,
            config=tiny_config(),
        )


def test_streaming_page_score_upper_bounds_every_page_logit():
    torch.manual_seed(0)
    page, pages = 16, 12
    cache = StreamingQuestCache(
        tiny_config(), max_length=512, device="cpu", dtype=torch.float32,
        sparse_budget=page, page_size=page, dense_layers=0,
        recent_tokens=0,
    )
    keys = torch.randn(1, KV_HEADS, pages * page, HEAD_DIM)
    cache.prefill_kv_cache(torch.randn_like(keys), 0, keys)
    query = torch.randn(1, HEADS, 1, HEAD_DIM)

    bound = cache._score_blocks(0, query, 0, pages)
    grouped_q = query.view(1, KV_HEADS, HEADS // KV_HEADS, 1, HEAD_DIM)
    paged_k = keys.view(1, KV_HEADS, pages, page, HEAD_DIM)
    true_max = torch.einsum(
        "bhgtd,bhpsd->bhgtps", grouped_q, paged_k
    ).amax(dim=-1).amax(dim=2).squeeze(2)
    assert torch.all(bound >= true_max - 1e-5)


def test_streaming_score_matches_quest_reference_sign_trick():
    torch.manual_seed(1)
    page, pages = 16, 8
    keys = torch.randn(1, KV_HEADS, pages * page, HEAD_DIM)
    query = torch.randn(1, KV_HEADS, 1, HEAD_DIM)

    sign = (query > 0).to(query.dtype) * 2 - 1
    chunk_max = (keys * sign).view(
        1, KV_HEADS, pages, page, HEAD_DIM
    ).amax(dim=-2)
    reference = torch.einsum(
        "bhqd,bhpd->bhqp", query * sign, chunk_max
    ).squeeze(2)

    paged = keys.view(1, KV_HEADS, pages, page, HEAD_DIM)
    page_min, page_max = paged.amin(dim=-2), paged.amax(dim=-2)
    ours = torch.maximum(
        query.unsqueeze(-2) * page_min.unsqueeze(2),
        query.unsqueeze(-2) * page_max.unsqueeze(2),
    ).sum(-1).squeeze(2)
    torch.testing.assert_close(ours, reference)
    assert torch.equal(ours.argsort(dim=-1), reference.argsort(dim=-1))


def test_short_context_attends_every_available_token_before_budget_fills():
    cache = StreamingQuestCache(
        tiny_config(), max_length=64, device="cpu", dtype=torch.float32,
        sparse_budget=16, page_size=8, dense_layers=0,
        prefix_tokens=8, recent_tokens=8,
    )
    keys = torch.randn(1, KV_HEADS, 8, HEAD_DIM)
    values = torch.randn_like(keys)
    cache.prefill_kv_cache(values, 0, keys)
    query = torch.randn(1, HEADS, 1, HEAD_DIM)
    selected_k, selected_v = cache.select_key_value_cache(0, query)
    torch.testing.assert_close(selected_k, keys)
    torch.testing.assert_close(selected_v, values)

    # At 24 tokens only one block is a retrieval candidate; prefix and recent
    # cover the other two.  The configured budget asks for two candidates, so
    # warm-up must select the sole candidate rather than fail or pad ids.
    more_k = torch.randn(1, KV_HEADS, 16, HEAD_DIM)
    more_v = torch.randn_like(more_k)
    cache.update_kv_cache(more_k, more_v, 0)
    selected_k, selected_v = cache.select_key_value_cache(0, query)
    assert selected_k.shape[-2] == 24
    assert selected_v.shape[-2] == 24


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("group_reduce", ["max", "sum"])
def test_fused_quest_router_matches_reference(group_reduce):
    torch.manual_seed(19)
    pages = 8192
    groups = HEADS // KV_HEADS
    query = torch.randn(
        1, KV_HEADS, groups, 1, HEAD_DIM,
        device="cuda", dtype=torch.bfloat16,
    )
    minimum = torch.randn(
        1, KV_HEADS, pages, HEAD_DIM,
        device="cuda", dtype=torch.bfloat16,
    )
    maximum = minimum + torch.rand_like(minimum).abs()
    actual = quest_page_scores(
        query, minimum, maximum,
        first_page=5, last_page=pages - 7,
        group_reduce=group_reduce,
    )

    q = query.float().unsqueeze(-2)
    reference = torch.maximum(
        q * minimum.float()[:, :, None, None, 5:-7],
        q * maximum.float()[:, :, None, None, 5:-7],
    ).sum(-1).squeeze(-2)
    reference = (
        reference.amax(dim=2)
        if group_reduce == "max"
        else reference.sum(dim=2)
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=2e-4)
    actual_top = actual.topk(256, -1).indices
    expected_top = reference.topk(256, -1).indices
    overlap = (
        actual_top[..., :, None] == expected_top[..., None, :]
    ).any(-1).float().mean()
    assert overlap.item() >= 0.999
