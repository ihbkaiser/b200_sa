from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ShadowKV"))

from models.exact_totalmass_rerank_cache import (  # noqa: E402
    StreamingExactTotalMassRerankCache,
)


def _cache(mode: str) -> StreamingExactTotalMassRerankCache:
    cache = StreamingExactTotalMassRerankCache.__new__(
        StreamingExactTotalMassRerankCache
    )
    cache.query_group_mean = mode == "qmean"
    cache.num_key_value_groups = 4
    cache.group_reduce = mode if mode != "qmean" else "sum"
    return cache


def test_exact_rerank_gqa_max_reduces_query_heads_after_query_sum() -> None:
    score = torch.arange(2 * 3 * 4 * 2 * 5, dtype=torch.float32).view(
        2, 3, 4, 2, 5
    )
    actual = _cache("max")._reduce_query_groups(score, query_dim=-2)
    expected = score.sum(dim=-2).amax(dim=2)
    torch.testing.assert_close(actual, expected)


def test_exact_rerank_gqa_sum_reduces_query_heads_after_query_sum() -> None:
    score = torch.arange(2 * 3 * 4 * 2 * 5, dtype=torch.float32).view(
        2, 3, 4, 2, 5
    )
    actual = _cache("sum")._reduce_query_groups(score, query_dim=-2)
    expected = score.sum(dim=-2).sum(dim=2)
    torch.testing.assert_close(actual, expected)


def test_exact_rerank_qmean_removes_singleton_group() -> None:
    score = torch.arange(2 * 3 * 1 * 1 * 5, dtype=torch.float32).view(
        2, 3, 1, 1, 5
    )
    actual = _cache("qmean")._reduce_query_groups(score, query_dim=-2)
    expected = score.sum(dim=-2).squeeze(2)
    torch.testing.assert_close(actual, expected)
