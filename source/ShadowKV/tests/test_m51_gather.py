"""
The optimised decode must not change the answer.

`decode_mode='dense_mask'` is the reference: it computes every logit and masks,
which is obviously correct and obviously slow. `decode_mode='gather'` reads only
the blocks it selects. With load_batch=1 the two walk the block order at the
same granularity and must agree exactly; with a larger batch the gather path
rounds its stopping depth UP to a multiple of the batch, so it attends to a
superset -- never a subset, which would silently lose mass.

  python -m pytest ShadowKV/tests/test_m51_gather.py -q
"""

import os
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.routed_pq_cache import RoutedPQCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

H, G, D, LAYERS, M = 4, 2, 64, 2, 4
HEADS = H * G


def cfg():
    return types.SimpleNamespace(hidden_size=HEADS * D, num_attention_heads=HEADS,
                                 num_key_value_heads=H, num_hidden_layers=LAYERS, head_dim=D)


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    """A small offline bank: anchors, calibration queries, and a PQ codebook."""
    rng = np.random.default_rng(0)
    path = tmp_path_factory.mktemp("m51") / "bank.npz"
    np.savez(
        path,
        anchors=rng.normal(size=(LAYERS, H, M, D)).astype(np.float32) * 0.1,
        calib=rng.normal(size=(LAYERS, H, G, 32, D)).astype(np.float16) * 0.1,
        pq_codebook=rng.normal(size=(LAYERS, H, D // 8, 64, 8)).astype(np.float32) * 0.1,
        n_anchors=M, dim=D, n_groups=G,
    )
    return str(path)


def build(bank, **kw):
    torch.manual_seed(0)
    c = RoutedPQCache(cfg(), max_length=2048, anchors_path=bank,
                      n_anchors=M, coverage=0.90, dense_layers=0, **kw)
    return c


def run(cache, T=1024, seed=0):
    torch.manual_seed(seed)
    keys = torch.randn(1, H, T, D, device="cuda", dtype=torch.bfloat16) * 0.3
    vals = torch.randn_like(keys)
    for layer in range(LAYERS):
        cache.prefill_kv_cache(vals, layer, keys)
    q = torch.randn(1, HEADS, 1, D, device="cuda", dtype=torch.bfloat16)
    new_k = torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16)
    outs = []
    for layer in range(LAYERS):
        cache.update_kv_cache(new_k, torch.randn_like(new_k), layer)
        outs.append(cache.decode_attend(layer, q))
    return torch.cat(outs), cache.traffic()


@CUDA
def test_gather_matches_the_dense_reference_at_batch_one(bank):
    ref, ref_traffic = run(build(bank, decode_mode="dense_mask", group_select="shared"))
    got, got_traffic = run(build(bank, decode_mode="gather", group_select="shared",
                                 load_batch=1))
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-3, atol=2e-3)
    assert abs(got_traffic["group_union_block_frac"]
               - ref_traffic["group_union_block_frac"]) < 1e-9


@CUDA
@pytest.mark.parametrize("batch", [8, 64])
def test_larger_batches_load_a_superset_never_a_subset(bank, batch):
    _, ref = run(build(bank, decode_mode="gather", group_select="shared", load_batch=1))
    _, got = run(build(bank, decode_mode="gather", group_select="shared", load_batch=batch))
    assert got["group_union_block_frac"] >= ref["group_union_block_frac"] - 1e-9


@CUDA
def test_shared_selection_makes_a_group_read_one_set(bank):
    """Under a shared order the group's loads are one prefix, so per-query
    traffic and union traffic are the same number.

    Note what is NOT asserted: that shared is always cheaper than the per-query
    union. It is not, and the reason is worth stating. Shared trades a union of
    different sets for a deeper common prefix, because the group-reduced order
    is nobody's own best order. Which side wins depends on how correlated the
    group's rankings are -- highly correlated on real context, uncorrelated on
    the random tensors used here, where shared in fact costs slightly more.
    That comparison is an empirical question for the RULER cells, not an
    invariant a unit test can pin.
    """
    _, shared = run(build(bank, decode_mode="dense_mask", group_select="shared"))
    assert abs(shared["per_query_block_frac"] - shared["group_union_block_frac"]) < 1e-9

    _, per_q = run(build(bank, decode_mode="dense_mask", group_select="per_query"))
    # per-query selection is the finer one, so each individual query reads no
    # more than it would under a shared order -- the cost shows up in the union
    assert per_q["per_query_block_frac"] <= shared["per_query_block_frac"] + 1e-9


@CUDA
def test_offline_codebook_is_required_when_asked_for(bank, tmp_path):
    z = dict(np.load(bank))
    z.pop("pq_codebook")
    p = tmp_path / "nocb.npz"
    np.savez(p, **z)
    with pytest.raises(ValueError, match="no offline PQ codebook"):
        build(str(p), pq_mode="offline")
    build(str(p), pq_mode="per_context")          # per_context needs no codebook


@CUDA
def test_per_query_gather_matches_its_dense_reference(bank):
    """The per-query gather path reads only what each query head selected.

    It must still produce the reference answer: at load_batch=1 it walks each
    group's own order one block at a time, exactly as dense_mask does before
    masking, so the two agree. Its traffic must also report per-query and union
    as DIFFERENT numbers -- under per-query selection the group's union is the
    read a shared KV cache pays, and collapsing the two hid that once already.
    """
    ref, ref_traffic = run(build(bank, decode_mode="dense_mask", group_select="per_query"))
    got, got_traffic = run(build(bank, decode_mode="gather", group_select="per_query",
                                 load_batch=1))
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        torch.tensor(got_traffic["per_query_block_frac"]),
        torch.tensor(ref_traffic["per_query_block_frac"]), rtol=0, atol=1e-9)
    assert got_traffic["group_union_block_frac"] >= got_traffic["per_query_block_frac"] - 1e-9


@CUDA
@pytest.mark.parametrize("budget", [128, 512])
def test_fixed_budget_reads_exactly_its_budget(bank, budget):
    """The point of select_mode='fixed' is that the budget IS the traffic.

    Under 'adaptive' the depth is decided per query head and the group pays the
    union, so the flag value and the tokens read are two different numbers --
    the same trap ShadowKV's budget flag sets, and the reason the campaign has
    a budget_audit step. Here they must coincide exactly.
    """
    cache = build(bank, select_mode="fixed", sparse_budget=budget)
    _, t = run(cache)
    expect = (budget // cache.leaf_size) / cache.total_blocks
    assert abs(t["group_union_block_frac"] - expect) < 1e-9
    assert abs(t["per_query_block_frac"] - expect) < 1e-9


@CUDA
def test_fixed_budget_attends_to_exactly_the_blocks_it_ranked(bank):
    """Output must equal an explicit softmax over the top-k blocks of the
    group-reduced bound, plus the trailing local tokens."""
    cache = build(bank, select_mode="fixed", sparse_budget=256)
    T = 1024
    torch.manual_seed(0)
    keys = torch.randn(1, H, T, D, device="cuda", dtype=torch.bfloat16) * 0.3
    vals = torch.randn_like(keys)
    for layer in range(LAYERS):
        cache.prefill_kv_cache(vals, layer, keys)
    q = torch.randn(1, HEADS, 1, D, device="cuda", dtype=torch.bfloat16)
    new_k, new_v = (torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16),) * 2
    layer = 0
    cache.update_kv_cache(new_k, new_v, layer)
    got = cache.decode_attend(layer, q)

    ls, L = cache.leaf_size, cache.n_leaves
    end = cache._end(layer)
    qs = q.view(H, G, D).float() * cache.scale
    upper = cache._upper_bounds(layer, qs)
    k = 256 // ls
    sel = upper.amax(dim=1).topk(k, dim=-1).indices
    tok = (sel.unsqueeze(-1) * ls + torch.arange(ls, device="cuda")).view(H, k * ls)
    tok = torch.cat([tok, torch.arange(L * ls, end, device="cuda").unsqueeze(0).expand(H, -1)], -1)
    kk = torch.gather(cache.k_cache[layer][0, :, :end], 1,
                      tok.unsqueeze(-1).expand(H, tok.shape[-1], D)).float()
    vv = torch.gather(cache.v_cache[layer][0, :, :end], 1,
                      tok.unsqueeze(-1).expand(H, tok.shape[-1], D)).float()
    p = torch.softmax(torch.einsum('hgd,hnd->hgn', qs, kk), dim=-1)
    ref = torch.einsum('hgn,hnd->hgd', p, vv).view(1, 1, H * G, D)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-3, atol=2e-3)


@CUDA
@pytest.mark.parametrize("n", [1, 2, 0])
def test_all_anchor_bound_is_the_min_over_the_branches_it_looks_at(bank, n):
    """min over anchors must be a real min of per-anchor bounds, and taking more
    anchors must never give a looser number -- that monotonicity is the whole
    argument for spending the extra gathers."""
    cache = build(bank, select_mode="fixed", sparse_budget=256,
                  rank_anchors=n, rank_reduce="max")
    T = 512
    torch.manual_seed(0)
    keys = torch.randn(1, H, T, D, device="cuda", dtype=torch.bfloat16) * 0.3
    for layer in range(LAYERS):
        cache.prefill_kv_cache(torch.randn_like(keys), layer, keys)
    q = torch.randn(H, G, D, device="cuda") * cache.scale

    got = (cache._upper_bounds(0, q) if n == 1 else cache._all_anchor_bounds(0, q, n))
    # explicit reference: score every anchor, keep the smallest
    L = cache.n_leaves
    lut = torch.einsum('hgsp,hscp->hgsc',
                       q.view(H, G, cache.n_sub, cache.pq_subdim), cache.codebook[0])
    per = []
    for m in range(M):
        sc = sum(lut[:, :, s, :].gather(2, cache.codes[0][:, s, m].long()
                                        .unsqueeze(1).expand(H, G, L))
                 for s in range(cache.n_sub))
        per.append(sc + cache.offsets[0][:, m].unsqueeze(1) + cache.eta[0][:, m].transpose(0, 1)
                   .transpose(0, 1))
    per = torch.stack(per, 2)                                     # [H,G,M,L]
    order = torch.cdist(q, cache.anchors[0]).argsort(dim=-1)      # nearest first
    take = M if n == 0 else n
    hh = torch.arange(H, device="cuda").unsqueeze(1)
    gg = torch.arange(G, device="cuda").unsqueeze(0)
    ref = torch.stack([per[hh, gg, order[..., i]] for i in range(take)], 2).amin(2)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    if n != 1:
        routed = cache._upper_bounds(0, q)
        assert bool((got <= routed + 1e-4).all()), "more anchors must not loosen the bound"


@CUDA
def test_warm_start_with_zero_steps_is_exactly_the_offline_codebook(bank):
    """pq_mode='warm' seeds k-means from the offline codebook, so at zero Lloyd
    steps it must reproduce 'offline' bit for bit. That pins the seeding: a warm
    start that silently re-initialised would look like a small accuracy change
    rather than a bug."""
    a = build(bank, select_mode="fixed", sparse_budget=256, pq_mode="offline")
    b = build(bank, select_mode="fixed", sparse_budget=256, pq_mode="warm",
              pq_warm_iters=0)
    out_a, _ = run(a)
    out_b, _ = run(b)
    torch.testing.assert_close(out_a.float(), out_b.float(), rtol=0, atol=0)
    torch.testing.assert_close(a.codebook, b.codebook, rtol=0, atol=0)
    assert torch.equal(a.codes, b.codes)


@CUDA
def test_warm_start_moves_the_codebook_and_still_decodes(bank):
    """And with steps it must actually differ -- otherwise 'warm' is a no-op
    wearing the name of a fix."""
    a = build(bank, select_mode="fixed", sparse_budget=256, pq_mode="offline")
    b = build(bank, select_mode="fixed", sparse_budget=256, pq_mode="warm",
              pq_warm_iters=2)
    run(a)
    out_b, traffic = run(b)
    assert not torch.allclose(a.codebook, b.codebook)
    assert torch.isfinite(out_b).all()
    expect = (256 // b.leaf_size) / b.total_blocks
    assert abs(traffic["group_union_block_frac"] - expect) < 1e-9


@CUDA
@pytest.mark.parametrize("codes", [257, 1024])
def test_pq_codes_beyond_uint8_is_refused(bank, codes):
    """Not a style check. pq_codes=1024 ran to completion and scored 0.00 on
    niah_multikey_3, because the uint8 code index wrapped and every block read a
    wrong codebook row. A silent wrong answer is worse than a crash."""
    with pytest.raises(AssertionError, match="uint8"):
        build(bank, select_mode="fixed", sparse_budget=256, pq_codes=codes)
