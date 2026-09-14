"""Gates for RetroInfer's author index and kernels on the common path.

The five properties a wrong wiring would break silently:

1. the frame partitions the context -- every token is either exact or
   clustered, never both, at every flush;
2. the store really is cluster-ordered, and its permutation agrees with the
   authors' own cluster sizes and value sums;
3. a budget that covers the whole indexed region reproduces dense attention,
   which pins the gather and the exact regions;
4. the authors' two ``weighted_flash_decoding`` calls, chained through
   ``previous_out``/``previous_lse``, equal one softmax over the union;
5. the estimation zone weights a cluster by its size -- the property that makes
   it an estimate of ``cluster_size`` keys rather than one key at the centroid.

Needs a CUDA device, the authors' Triton k-means, their compiled
``retroinfer_kernels`` and their ``weighted_flash_decoding`` fork.
"""

from pathlib import Path
from types import SimpleNamespace
import os
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ShadowKV"))

from models.retroinfer_author_streaming_cache import (  # noqa: E402
    StreamingRetroInferAuthorCache,
)

AUTHOR_ROOT = os.environ.get(
    "RETROINFER_AUTHOR_ROOT", "/home/baonn/upstream-kv-methods/RetrievalAttention"
)
DEVICE = "cuda:0"
DTYPE = torch.bfloat16


# head_dim 128: the authors' copy kernels fix the vector length there.
def _config(layers=2, heads=8, kv_heads=2, head_dim=128):
    return SimpleNamespace(
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        num_hidden_layers=layers,
        head_dim=head_dim,
        hidden_size=heads * head_dim,
    )


def _cache(*, sparse_budget=256, prefix=32, recent=64, interval=64,
           block=8, estimation_ratio=0.232, average_cluster_size=8,
           max_length=4096, offload=False):
    config = _config()
    return StreamingRetroInferAuthorCache(
        config,
        author_root=AUTHOR_ROOT,
        max_length=max_length,
        device=DEVICE,
        dtype=DTYPE,
        batch_size=1,
        sparse_budget=sparse_budget,
        block_size=block,
        prefix_tokens=prefix,
        recent_tokens=recent,
        update_interval=interval,
        offload=offload,
        average_cluster_size=average_cluster_size,
        n_segment=4,
        estimation_ratio=estimation_ratio,
    )


def _random_kv(cache, tokens, generator):
    shape = (
        cache.batch_size,
        cache.num_key_value_heads,
        tokens,
        cache.head_dim,
    )
    keys = torch.randn(shape, device=DEVICE, dtype=torch.float32,
                       generator=generator).to(DTYPE)
    values = torch.randn(shape, device=DEVICE, dtype=torch.float32,
                         generator=generator).to(DTYPE)
    return keys, values


def test_frame_partitions_the_context_at_every_flush() -> None:
    cache = _cache()
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    prompt = 1024
    keys, values = _random_kv(cache, prompt, generator)
    for layer in range(cache.num_layers):
        cache.prefill_kv_cache(values, layer, keys)

    for step in range(200):
        step_k, step_v = _random_kv(cache, 1, generator)
        for layer in range(cache.num_layers):
            cache.update_kv_cache(step_k, step_v, layer)
            state = cache.block_state[layer]
            active_end = state.active_blocks * cache.block_size
            # The index reaches exactly as far as retrieval is allowed to.
            assert cache.indexed_end[layer] == max(active_end, cache.prefix_tokens), (
                f"layer {layer} step {step}: index ends at "
                f"{cache.indexed_end[layer]}, candidates end at {active_end}"
            )
            covered = sorted(state.exact_ranges)
            expected = [(0, cache.prefix_tokens)]
            if active_end < state.total_tokens:
                expected.append((active_end, state.total_tokens))
            assert covered == expected, (
                f"layer {layer} step {step}: exact ranges {covered} do not "
                f"complement the indexed region [{cache.prefix_tokens},"
                f"{active_end})"
            )


def test_a_shorter_prompt_does_not_inherit_the_previous_index() -> None:
    # The cluster axis is scored at its full width, so the liveness mask is the
    # only thing separating this prompt's clusters from the last one's.  A long
    # prompt followed by a short one used to leave the tail marked live, and the
    # router then selected slots holding the previous prompt's tokens.
    cache = _cache()
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    long_keys, long_values = _random_kv(cache, 2048, generator)
    cache.prefill_kv_cache(long_values, 0, long_keys)
    grown = cache.cluster_count[0]
    assert grown > 0

    cache.clear()
    short_keys, short_values = _random_kv(cache, 512, generator)
    cache.prefill_kv_cache(short_values, 0, short_keys)
    now = cache.cluster_count[0]
    assert 0 < now < grown, f"test needs a smaller index: {now} vs {grown}"

    live = ~cache.empty_cluster[0]
    assert not bool(live[..., now:].any()), (
        "clusters past the new index are still marked live -- the next prompt "
        "would retrieve the previous prompt's tokens"
    )
    assert int(cache.cluster_pages[0, :, :, now:].sum()) == 0


def test_cluster_order_survives_one_degenerate_cluster() -> None:
    # The authors' padded inverted list is sized by the largest cluster, so a
    # single cluster holding most of the segment made it ask for 19 GiB and
    # killed a 128K cell.  Ordering by sort must not care how uneven the
    # clustering is.
    cache = _cache()
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    tokens, count = 4096, 64
    keys = torch.randn(cache.batch_groups, tokens, cache.head_dim, device=DEVICE,
                       dtype=torch.float32, generator=generator).to(DTYPE)
    # Collapse almost every key onto one point: k-means then has no choice but
    # to put them in one cluster.
    keys[:, 16:] = keys[:, :1]
    values = torch.randn_like(keys, dtype=torch.float32).to(DTYPE)

    centroids, value_sum, order, owners, sizes = cache._cluster_segment(
        keys, values, count, 1
    )
    assert int(sizes.amax()) > tokens // 2, "the test did not create a big cluster"
    for g in range(cache.batch_groups):
        assert sorted(order[g].tolist()) == list(range(tokens)), "not a permutation"
        run = owners[g]
        assert bool((run[1:] >= run[:-1]).all()), "slots are not cluster-ordered"
    torch.testing.assert_close(
        sizes.sum(-1), torch.full_like(sizes.sum(-1), tokens)
    )
    assert centroids.shape[1] == count and value_sum.shape[1] == count


def test_cluster_ordered_store_matches_the_author_cluster_statistics() -> None:
    cache = _cache()
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    prompt = 1024
    keys, values = _random_kv(cache, prompt, generator)
    cache.prefill_kv_cache(values, 0, keys)

    count = cache.cluster_count[0]
    fill = cache.index_fill[0]
    assert count > 0
    start, end = cache.prefix_tokens, cache.indexed_end[0]
    assert fill == end - start
    owner = cache.slot_cluster[0, 0, :, :fill].long()
    sizes = cache.cluster_size[0, 0, :, :count]
    for head in range(cache.num_key_value_heads):
        observed = torch.bincount(owner[head], minlength=count)
        torch.testing.assert_close(observed.float(), sizes[head].float())
        # Cluster order means each cluster owns one contiguous run of slots.
        run = owner[head]
        assert bool((run[1:] >= run[:-1]).all()), "slots are not cluster-ordered"

    # The store holds exactly the indexed tokens, permuted in place.
    assert cache.index_origin == cache.prefix_tokens
    first = cache.index_origin
    stored = cache.k_cache[0, :, :, first : first + fill].to(DEVICE).float()
    source = keys[0, :, start:end].float()
    order_s, _ = stored[0].sort(dim=-2)
    order_c, _ = source.sort(dim=-2)
    torch.testing.assert_close(order_s, order_c)

    # value_sum is the sum of the member values, which is what the estimation
    # zone divides by cluster_size to stand in for the real ones.
    member_values = values[0, :, start:end].float()
    rebuilt = torch.zeros(
        cache.num_key_value_heads, count, cache.head_dim, device=DEVICE
    )
    ordered_values = cache.v_cache[0, 0, :, first : first + fill].to(DEVICE).float()
    rebuilt.scatter_add_(
        1,
        owner.unsqueeze(-1).expand(-1, -1, cache.head_dim),
        ordered_values,
    )
    stored_sum = cache.value_sum[0, 0, :, :count].float()
    torch.testing.assert_close(rebuilt, stored_sum, rtol=3e-2, atol=3e-2)
    assert member_values.shape == ordered_values.shape


def test_full_budget_retrieval_reproduces_dense_attention() -> None:
    # A budget wider than the indexed region leaves nothing to estimate, so
    # the method degenerates to dense attention over the whole context.
    cache = _cache(sparse_budget=2048, estimation_ratio=0.0)
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    prompt = 1024
    keys, values = _random_kv(cache, prompt, generator)
    cache.prefill_kv_cache(values, 0, keys)
    step_k, step_v = _random_kv(cache, 1, generator)
    cache.update_kv_cache(step_k, step_v, 0)

    query = torch.randn(
        cache.batch_size,
        cache.num_attention_heads,
        1,
        cache.head_dim,
        device=DEVICE,
        dtype=torch.float32,
        generator=generator,
    ).to(DTYPE)
    gathered_k, gathered_v = cache.select_key_value_cache(0, query)
    assert cache._pending_estimation[0] is None
    total = cache.block_state[0].total_tokens
    # Retrieval now reads the cluster-ordered region and the exact regions come
    # from the chronological one; together they must be the whole context, each
    # token once.
    assert gathered_k.shape[-2] == total, (
        f"gathered {gathered_k.shape[-2]} of {total} tokens"
    )
    expected_k = cache.k_cache[0, :, :, :total].to(DEVICE)
    order_k, _ = gathered_k.float().sort(dim=-2)
    order_e, _ = expected_k.float().sort(dim=-2)
    torch.testing.assert_close(order_k, order_e)
    assert gathered_v.shape == gathered_k.shape


def test_author_gemm_softmax_scores_clusters_like_the_paper() -> None:
    # Their fused CUTLASS scorer must equal softmax(Q C^T / sqrt(d)) summed over
    # the query heads that share a KV head, which is what their sparse_attention
    # computes as batch_gemm_softmax + sum(softmax_o, dim=1).  The cluster axis
    # is a GEMM dimension, so this also pins its alignment: an unaligned width
    # faults at run time rather than failing the launch.
    cache = _cache()
    groups = cache.batch_groups
    heads = cache.num_key_value_groups
    dim = cache.head_dim
    width = cache.max_clusters
    assert width % cache.centroid_multiple == 0, "cluster axis is not aligned"
    generator = torch.Generator(device=DEVICE).manual_seed(6)
    live = 64

    centroids = torch.randn(1, cache.num_key_value_heads, width, dim,
                            device=DEVICE, dtype=torch.float32,
                            generator=generator).to(DTYPE)
    cache.centroids[0] = centroids
    cache.cluster_size[0, :, :, :live] = 1
    cache.empty_cluster[0, :, :, :live] = False
    cache.cluster_count[0] = live
    cache._ensure_scratch(width, width, 1)

    query = torch.randn(1, cache.num_attention_heads, 1, dim, device=DEVICE,
                        dtype=torch.float32, generator=generator).to(DTYPE)
    queries = query[:, :, -1:].transpose(1, 2).contiguous()
    distance = cache._cluster_scores(0, queries, width)

    reference = torch.einsum(
        "bhgd,bhcd->bhgc",
        query.view(1, cache.num_key_value_heads, heads, dim).float(),
        centroids.float(),
    ) * cache.scale
    reference = reference.softmax(-1).sum(2).view(groups, width)
    torch.testing.assert_close(
        distance[:, :live].float(), reference[:, :live], rtol=2e-2, atol=2e-2
    )
    # Clusters that hold nothing must never be ranked.
    assert bool((distance[:, live:] == cache.dtype_min).all())


def test_author_attention_equals_one_softmax_over_the_union() -> None:
    # A zone of size-one clusters is exact: each cluster then stands for its own
    # single key.  The authors' two weighted_flash_decoding calls must therefore
    # reproduce a plain softmax over the retrieved rows and the zone together --
    # that is what previous_out/previous_lse is for.
    cache = _cache(estimation_ratio=0.0)
    heads = cache.num_attention_heads
    kv_heads = cache.num_key_value_heads
    groups = cache.num_key_value_groups
    dim = cache.head_dim
    generator = torch.Generator(device=DEVICE).manual_seed(3)

    estimated, retrieved, count = 12, 24, 16
    query = torch.randn(1, heads, 1, dim, device=DEVICE, dtype=torch.float32,
                        generator=generator).to(DTYPE)
    centroids = torch.randn(1, kv_heads, count, dim, device=DEVICE,
                            dtype=torch.float32, generator=generator).to(DTYPE)
    value_sum = torch.randn(1, kv_heads, count, dim, device=DEVICE,
                            dtype=torch.float32, generator=generator).to(DTYPE)
    keys = torch.randn(1, kv_heads, retrieved, dim, device=DEVICE,
                       dtype=torch.float32, generator=generator).to(DTYPE)
    values = torch.randn(1, kv_heads, retrieved, dim, device=DEVICE,
                         dtype=torch.float32, generator=generator).to(DTYPE)

    cache.centroids[0, :, :, :count] = centroids
    cache.value_sum[0, :, :, :count] = value_sum
    cache.cluster_size[0, :, :, :count] = 1
    cache._ensure_scratch(cache.max_clusters, cache.max_clusters, estimated)
    cache._author_queries[0] = query.transpose(1, 2).contiguous()
    cache._estimation_ids[0] = torch.arange(
        estimated, device=DEVICE
    ).view(1, -1).expand(cache.batch_groups, -1).contiguous()
    cache._pending_estimation[0] = cache.max_clusters

    merged = cache.decode_attend_gathered(0, query, keys, values)

    scale = cache.scale
    q = query.view(1, kv_heads, groups, dim).float()
    union_k = torch.cat((keys.float(), centroids[:, :, :estimated].float()), 2)
    union_v = torch.cat((values.float(), value_sum[:, :, :estimated].float()), 2)
    logits = torch.einsum("bhgd,bhnd->bhgn", q, union_k) * scale
    expected = (logits.softmax(-1) @ union_v).reshape(1, heads, dim)
    torch.testing.assert_close(
        merged.float().reshape(1, heads, dim), expected, rtol=8e-3, atol=8e-3
    )


def test_estimation_zone_weights_a_cluster_by_its_size() -> None:
    # The zone's defining property: a cluster of n tokens enters the softmax
    # with denominator weight n and numerator value_sum, so it is not the same
    # as attending one key at the centroid.
    cache = _cache(estimation_ratio=0.0)
    kv_heads = cache.num_key_value_heads
    groups = cache.num_key_value_groups
    dim = cache.head_dim
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    count = 8

    query = torch.randn(cache.batch_groups, 1, groups, dim, device=DEVICE,
                        dtype=torch.float32, generator=generator).to(DTYPE)
    centroids = torch.randn(cache.batch_groups, count, 1, dim, device=DEVICE,
                            dtype=torch.float32, generator=generator).to(DTYPE)
    value_sum = torch.randn(cache.batch_groups, count, 1, dim, device=DEVICE,
                            dtype=torch.float32, generator=generator).to(DTYPE)
    sizes = torch.randint(1, 30, (cache.batch_groups, 1, 1, count),
                          device=DEVICE).to(DTYPE)

    out, lse = cache._weighted_flash_decoding(
        query, centroids, value_sum, sizes,
        previous_out=None, previous_lse=None, return_softmax_lse=True,
    )

    scale = cache.scale
    logits = torch.einsum(
        "bgd,bnd->bgn", query.squeeze(1).float(), centroids.squeeze(2).float()
    ) * scale
    weights = (logits - logits.amax(-1, keepdim=True)).exp()
    size = sizes.view(cache.batch_groups, 1, count).float()
    expected = (weights @ value_sum.squeeze(2).float()) / (
        weights * size
    ).sum(-1, keepdim=True)
    torch.testing.assert_close(
        out.squeeze(1).float(), expected, rtol=8e-3, atol=8e-3
    )
    del lse


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
