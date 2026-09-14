from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ShadowKV"))

from models.pqcache_author_native_cache import PQCacheAuthorNativeCache


def test_native_ratio_encoding_preserves_explicit_token_accounting():
    cache = object.__new__(PQCacheAuthorNativeCache)
    cache.sink_tokens = 32
    cache.recent_tokens = 32
    for length, budget in ((32768, 1024), (65536, 2048), (131072, 4096)):
        cache.retrieved_tokens = budget
        ratio, local_ratio = cache._ratios_for_length(length)
        available = length - cache.sink_tokens
        assert int(available * ratio * local_ratio) == 32
        assert int(available * ratio * (1.0 - local_ratio)) == budget


def test_native_ratio_encoding_rejects_budget_larger_than_context():
    cache = object.__new__(PQCacheAuthorNativeCache)
    cache.sink_tokens = 32
    cache.recent_tokens = 32
    cache.retrieved_tokens = 1024
    try:
        cache._ratios_for_length(1000)
    except ValueError as exc:
        assert "exceeds context" in str(exc)
    else:
        raise AssertionError("oversized budget should fail")
