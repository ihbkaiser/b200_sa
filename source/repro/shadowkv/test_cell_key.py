"""
A cell's name must be injective: two configurations that produce different
numbers must never produce the same name, and a knob a method ignores must not
appear in its name (it would split one cell into several under different names).

  python -m pytest repro/shadowkv/test_cell_key.py -q
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_key import cell_key
from outlier_policy import shadow_outlier_chunks


def test_paper_outlier_rate_scales_with_context(monkeypatch):
    monkeypatch.delenv("SHADOWKV_OUTLIER_CHUNKS", raising=False)
    assert [shadow_outlier_chunks(n, 8) for n in
            (4096, 8192, 16384, 32768, 65536, 131072)] == [2, 3, 6, 12, 24, 48]


def test_outlier_override_changes_shadowkv_cell_name(monkeypatch):
    args = ("qwen3", 16384, "vt", "shadowkv", 512, 160, 8, 16, 2)
    monkeypatch.delenv("SHADOWKV_OUTLIER_CHUNKS", raising=False)
    assert cell_key(*args).endswith("_o6")
    monkeypatch.setenv("SHADOWKV_OUTLIER_CHUNKS", "48")
    assert cell_key(*args).endswith("_o48")


def test_adaptive_centroid_method_has_explicit_zero_outliers(monkeypatch):
    monkeypatch.setenv("SHADOWKV_OUTLIER_CHUNKS", "48")
    key = cell_key(
        "qwen3", 16384, "vt", "adaptive_centroid_lse",
        512, 160, 8, 16, 2,
    )
    assert key.endswith("adaptive_lse_b512_c8_x0.25_t1_o0_exactkv")


def test_adaptive_prefix4_has_a_distinct_explicit_name(monkeypatch):
    monkeypatch.setenv("SHADOWKV_OUTLIER_CHUNKS", "48")
    key = cell_key(
        "qwen3", 16384, "vt", "adaptive_centroid_lse_prefix4",
        512, 160, 8, 16, 2,
    )
    assert key.endswith(
        "adaptive_lse_prefix4_b512_c8_x0.25_t1_o0_exactkv"
    )


def test_mean_gap_minimax_configuration_has_an_injective_name(monkeypatch):
    monkeypatch.setenv("SHADOWKV_SELF_LSE_COST", "mean_gap")
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "self")
    monkeypatch.setenv("SHADOWKV_CENTER_ALLOCATION", "self_lse")
    key = cell_key(
        "qwen3", 32768, "cwe",
        "adaptive_centroid_lse_streaming_prefix4_querymean",
        1024, 160, 8, 16, 0, "max",
    )
    assert "_costmean_gap_" in key
    assert "_alloctail_cvar" not in key


def test_streaming_names_stamp_recent_window_and_exact_backing(monkeypatch):
    monkeypatch.setenv("STREAMING_RECENT_TOKENS", "32")
    quest = cell_key(
        "qwen3", 16384, "vt", "quest_streaming",
        512, 160, 8, 16, 2, "max",
    )
    ours = cell_key(
        "qwen3", 16384, "vt", "adaptive_centroid_lse_streaming_prefix4",
        512, 160, 8, 16, 2, "max",
    )
    assert quest.endswith("quest_stream_b512_p16_d2_max_x0_l32")
    assert ours.endswith(
        "adaptive_lse_stream_prefix4_b512_c8_x0.25_t1_max_l32_rmax8_rbtriton_exactkv"
    )

    ours_sum = cell_key(
        "qwen3", 16384, "vt", "adaptive_centroid_lse_streaming_prefix4",
        512, 160, 8, 16, 2, "sum",
    )
    assert ours_sum.endswith(
        "adaptive_lse_stream_prefix4_b512_c8_x0.25_t1_sum_l32_rmax8_rbtriton_exactkv"
    )
    monkeypatch.setenv("STREAMING_REFINE_FACTOR", "2")
    ours_refined = cell_key(
        "qwen3", 16384, "vt", "adaptive_centroid_lse_streaming_prefix4",
        512, 160, 8, 16, 2, "max",
    )
    assert ours_refined.endswith(
        "adaptive_lse_stream_prefix4_b512_c8_x0.25_t1_max_l32_rf2_rmax8_rbtriton_exactkv"
    )
    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "angular")
    ours_angular = cell_key(
        "qwen3", 16384, "vt", "adaptive_centroid_lse_streaming_prefix4",
        512, 160, 8, 16, 2, "max",
    )
    assert ours_angular.endswith(
        "adaptive_lse_stream_prefix4_b512_c8_x0.25_t1_max_l32_placeangular_rf2_rmax8_rbtriton_exactkv"
    )
    monkeypatch.delenv("SHADOWKV_CENTER_PLACEMENT")
    monkeypatch.delenv("STREAMING_REFINE_FACTOR")
    monkeypatch.setenv("STREAMING_REFINE_CANDIDATE_RATIO", "0.1")
    ours_ratio_refined = cell_key(
        "qwen3", 32768, "cwe", "adaptive_centroid_lse_streaming_prefix4",
        512, 160, 8, 16, 0, "max",
    )
    assert ours_ratio_refined.endswith(
        "adaptive_lse_stream_prefix4_b512_c8_x0.25_t1_max_l32_rr0.1_rmax8_rbtriton_exactkv"
    )
    monkeypatch.setenv("STREAMING_REFINE_TOKENS", "1")
    ours_token_refined = cell_key(
        "qwen3", 32768, "cwe", "adaptive_centroid_lse_streaming_prefix4",
        512, 160, 8, 16, 0, "max",
    )
    assert ours_token_refined.endswith(
        "adaptive_lse_stream_prefix4_b512_c8_x0.25_t1_max_l32_rr0.1_rtok_rmax8_rbtriton_exactkv"
    )
    monkeypatch.delenv("STREAMING_REFINE_TOKENS")
    monkeypatch.setenv("STREAMING_MAX_COMPONENTS", "8")
    ours_multilevel = cell_key(
        "qwen3", 32768, "cwe", "adaptive_centroid_lse_streaming_prefix4",
        512, 160, 8, 16, 0, "max",
    )
    assert ours_multilevel.endswith(
        "adaptive_lse_stream_prefix4_b512_c8_x0.25_t1_max_l32_rr0.1_rmax8_rbtriton_exactkv"
    )
    monkeypatch.delenv("STREAMING_MAX_COMPONENTS")
    monkeypatch.delenv("STREAMING_REFINE_CANDIDATE_RATIO")
    monkeypatch.setenv("STREAMING_COMPACT_METADATA", "1")
    monkeypatch.setenv("STREAMING_CENTER_BITS", "8")
    ours_compact_int8 = cell_key(
        "qwen3", 16384, "vt", "adaptive_centroid_lse_streaming_prefix4",
        512, 160, 8, 16, 2, "max",
    )
    assert ours_compact_int8.endswith(
        "adaptive_lse_stream_prefix4_b512_c8_x0.25_t1_max_l32_rmax8_compact_cb8_rbtriton_exactkv"
    )
    monkeypatch.delenv("STREAMING_COMPACT_METADATA")
    monkeypatch.delenv("STREAMING_CENTER_BITS")

    # Paris-style GQA averaging is performed before routing, so max/sum is no
    # longer a live knob.  It is a distinct experiment, but must honor the
    # same component budget as the non-averaged comparison.
    monkeypatch.setenv("ADAPTIVE_LSE_EXTRA_FRACTION", "3")
    ours_querymean = cell_key(
        "qwen3", 16384, "vt",
        "adaptive_centroid_lse_streaming_prefix4_querymean",
        512, 160, 8, 16, 2, "sum",
    )
    assert ours_querymean.endswith(
        "adaptive_lse_stream_prefix4_b512_c8_x3_t1_qmean_l32_"
        "alloctail_absolute_rate_distortion_tail0.25_ard1.5_"
        "costmean_gap_noalpha_"
        "rmax8_compact_cb8_rbtriton_exactkv"
    )

    # Bundle-aware allocation must encode the gap correction too.  Otherwise
    # gamma sweeps alias the same result file and concurrent workers corrupt it.
    monkeypatch.setenv("SHADOWKV_CENTER_ALLOCATION", "tail_rate_distortion")
    monkeypatch.setenv("SHADOWKV_TAIL_GAP_CORRECTION_SCALE", "0.25")
    bundle_g025 = cell_key(
        "qwen3", 32768, "cwe",
        "adaptive_centroid_lse_streaming_prefix4_querymean",
        512, 160, 8, 16, 0, "max",
    )
    monkeypatch.setenv("SHADOWKV_TAIL_GAP_CORRECTION_SCALE", "1")
    bundle_g1 = cell_key(
        "qwen3", 32768, "cwe",
        "adaptive_centroid_lse_streaming_prefix4_querymean",
        512, 160, 8, 16, 0, "max",
    )
    assert "_alloctail_rate_distortion_" in bundle_g025
    assert "_gapcorr0.25_" in bundle_g025
    assert "_gapcorr1_" in bundle_g1
    assert bundle_g025 != bundle_g1

    monkeypatch.setenv(
        "SHADOWKV_CENTER_ALLOCATION", "tail_absolute_rate_distortion"
    )
    monkeypatch.setenv("SHADOWKV_ABSOLUTE_RD_PENALTY", "0.03125")
    absolute_rd = cell_key(
        "qwen3", 32768, "cwe",
        "adaptive_centroid_lse_streaming_prefix4_querymean",
        512, 160, 8, 16, 0, "max",
    )
    assert "_alloctail_absolute_rate_distortion_" in absolute_rd
    assert "_ard0.03125_" in absolute_rd
    monkeypatch.delenv("SHADOWKV_ABSOLUTE_RD_PENALTY")
    monkeypatch.delenv("SHADOWKV_CENTER_ALLOCATION")
    monkeypatch.delenv("SHADOWKV_TAIL_GAP_CORRECTION_SCALE")
    monkeypatch.setenv("ADAPTIVE_LSE_EXTRA_FRACTION", "0.25")

    monkeypatch.setenv("SHADOWKV_CENTER_PLACEMENT", "qmass_bank")
    monkeypatch.setenv("SHADOWKV_CENTER_ALLOCATION", "marginal")
    monkeypatch.setenv("SHADOWKV_QUERY_MASS_BANK_SIZE", "32")
    monkeypatch.setenv("SHADOWKV_QUERY_MASS_WINDOW", "512")
    ours_qmass = cell_key(
        "qwen3", 32768, "cwe",
        "adaptive_centroid_lse_streaming_prefix4_querymean",
        512, 160, 8, 16, 0, "max",
    )
    assert "_placeqmass_bank32w512_allocmarginal_" in ours_qmass
    monkeypatch.setenv("SHADOWKV_QUERY_MASS_BANK_SIZE", "16")
    assert cell_key(
        "qwen3", 32768, "cwe",
        "adaptive_centroid_lse_streaming_prefix4_querymean",
        512, 160, 8, 16, 0, "max",
    ) != ours_qmass
    monkeypatch.setenv("SHADOWKV_QUERY_MASS_BANK_SIZE", "32")
    monkeypatch.setenv("SHADOWKV_QUERY_MASS_OBJECTIVE", "topk_hinge")
    ours_topk = cell_key(
        "qwen3", 32768, "cwe",
        "adaptive_centroid_lse_streaming_prefix4_querymean",
        512, 160, 8, 16, 0, "max",
    )
    assert "_placeqmass_bank32w512_topk_hinge_" in ours_topk
    monkeypatch.setenv("SHADOWKV_QUERY_MASS_GQA_REDUCE", "all")
    ours_gqa_all = cell_key(
        "qwen3", 32768, "cwe",
        "adaptive_centroid_lse_streaming_prefix4_querymean",
        512, 160, 8, 16, 0, "max",
    )
    assert "_topk_hinge_gqaall_" in ours_gqa_all
    monkeypatch.delenv("SHADOWKV_CENTER_PLACEMENT")
    monkeypatch.delenv("SHADOWKV_CENTER_ALLOCATION")
    monkeypatch.delenv("SHADOWKV_QUERY_MASS_BANK_SIZE")
    monkeypatch.delenv("SHADOWKV_QUERY_MASS_WINDOW")
    monkeypatch.delenv("SHADOWKV_QUERY_MASS_OBJECTIVE")
    monkeypatch.delenv("SHADOWKV_QUERY_MASS_GQA_REDUCE")

    monkeypatch.setenv("QUEST_PREFIX_TOKENS", "32")
    ours_fixed_prefix = cell_key(
        "qwen3", 16384, "vt", "adaptive_centroid_lse_streaming_prefix4",
        512, 160, 2, 16, 2, "max",
    )
    assert ours_fixed_prefix.endswith(
        "adaptive_lse_stream_p32_b512_c2_x0.25_t1_max_l32_rmax2_rbtriton_exactkv"
    )
    monkeypatch.delenv("QUEST_PREFIX_TOKENS")
    paris = cell_key(
        "qwen3", 16384, "vt", "pariskv_official",
        512, 160, 8, 16, 2, "max",
    )
    assert paris.endswith(
        "pariskv_official_b512_x32_l32_polarann"
    )
    paris_common = cell_key(
        "qwen3", 16384, "vt", "pariskv_author_common",
        512, 160, 8, 16, 2, "max",
    )
    assert paris_common.endswith(
        "pariskv_author_common_b512_c8_x32_l32_polarann"
    )
    retro = cell_key(
        "qwen3", 16384, "vt", "retroinfer_reference_streaming",
        512, 160, 8, 16, 2, "max",
    )
    assert retro.endswith(
        "retroinfer_ref_stream_b512_u1024_a16_e0.232_i10_x4_l64_exactkv"
    )
    monkeypatch.setenv("QUEST_PREFIX_TOKENS", "32")
    exact_lse = cell_key(
        "qwen3", 32768, "vt", "exact_block_lse_streaming",
        512, 160, 8, 16, 0, "max",
    )
    exact_max = cell_key(
        "qwen3", 32768, "vt", "exact_block_max_streaming",
        512, 160, 8, 16, 0, "max",
    )
    assert exact_lse.endswith(
        "exact_block_lse_stream_b512_c8_max_x32_l32_oracle"
    )
    assert exact_max.endswith(
        "exact_block_max_stream_b512_c8_max_x32_l32_oracle"
    )
    exact_lse_softmax = cell_key(
        "qwen3", 32768, "vt", "exact_block_lse_softmax_streaming",
        512, 160, 8, 16, 0, "max",
    )
    exact_max_softmax = cell_key(
        "qwen3", 32768, "vt", "exact_block_max_softmax_streaming",
        512, 160, 8, 16, 0, "max",
    )
    assert exact_lse_softmax.endswith(
        "exact_block_lse_softmax_stream_b512_c8_max_x32_l32_oracle"
    )
    assert exact_max_softmax.endswith(
        "exact_block_max_softmax_stream_b512_c8_max_x32_l32_oracle"
    )


def test_upstream_matched_exact_regions_have_injective_names(monkeypatch):
    monkeypatch.setenv("UPSTREAM_MATCHED_EXACT_REGIONS", "1")
    monkeypatch.setenv("QUEST_PREFIX_TOKENS", "32")
    monkeypatch.setenv("STREAMING_RECENT_TOKENS", "32")
    common = ("qwen3", 32768, "cwe")

    pq = cell_key(
        *common, "pqcache_author_common", 1024, 160, 8, 8, 0, "max"
    )
    magic = cell_key(
        *common, "magicpig_author_common", 1024, 160, 8, 8, 0, "max"
    )
    inf = cell_key(
        *common, "infllm_author_common", 1024, 160, 8, 8, 0, "max"
    )

    assert pq.endswith(
        "pqcache_author_common_routeb1024_pq2x6_i10_x32_l32_"
        "s4321_exactkv_matched"
    )
    assert magic.endswith(
        "magicpig_author_common_targetb1024_k10l210_x32_l32_d0_s43_matched"
    )
    assert inf.endswith(
        "infllm_author_common_routeb1024_x32_l32_blk128_repr4_matched"
    )

    monkeypatch.setenv("STREAMING_RECENT_TOKENS", "64")
    assert cell_key(
        *common, "pqcache_author_common", 1024, 160, 8, 8, 0, "max"
    ) != pq


def test_distinct_configs_get_distinct_names():
    seen = {}
    grid = itertools.product(
        ["llama32", "qwen3"], [8192, 16384], ["niah_single_1", "vt"],
        [("full", 0, 160, 8, 16, 2, "max")]
        + [("quest_streaming", b, 160, 8, p, d, g)
           for b in (512, 1024) for p in (16, 32) for d in (0, 2) for g in ("max", "sum")]
        + [("shadowkv", b, r, c, 16, 2, "max")
           for b in (512, 1024) for r in (160, 320) for c in (8, 16)],
    )
    for model, datalen, task, (method, budget, rank, chunk, page, dense, group) in grid:
        key = cell_key(model, datalen, task, method, budget, rank, chunk, page, dense, group)
        config = (model, datalen, task, method) + (
            () if method == "full"
            else (budget, page, dense, group) if method == "quest_streaming"
            else (budget, rank, chunk))
        if key in seen:
            assert seen[key] == config, f"name collision on {key}: {seen[key]} vs {config}"
        seen[key] = config


def test_querymean_literal_defaults_are_the_validated_self_k_preset(monkeypatch):
    for name in (
        "ADAPTIVE_LSE_EXTRA_FRACTION",
        "ADAPTIVE_LSE_TEMPERATURES",
        "SHADOWKV_CENTER_PLACEMENT",
        "SHADOWKV_CENTER_ALLOCATION",
        "SHADOWKV_ABSOLUTE_RD_PENALTY",
        "SHADOWKV_SELF_LSE_COST",
        "SHADOWKV_TAIL_CVAR_FRACTION",
        "SHADOWKV_TAIL_GAP_CORRECTION_SCALE",
        "SHADOWKV_CENTER_DISPERSION_CORRECTION",
        "STREAMING_MAX_COMPONENTS",
        "STREAMING_COMPACT_METADATA",
        "STREAMING_CENTER_BITS",
        "STREAMING_REFINE_FACTOR",
        "STREAMING_REFINE_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    key = cell_key(
        "qwen3", 32768, "cwe",
        "adaptive_centroid_lse_streaming_prefix4_querymean",
        1024, 160, 8, 16, 0, "max",
    )
    assert key.endswith(
        "adaptive_lse_stream_prefix4_b1024_c8_x0_t1_qmean_l32_"
        "alloctail_absolute_rate_distortion_tail0.25_ard1.5_"
        "costmean_gap_noalpha_"
        "rmax8_compact_cb8_rbtriton_exactkv"
    )


def test_ignored_knobs_stay_out_of_the_name():
    # streaming Quest does not read rank/chunk; ShadowKV ignores page/dense
    assert (cell_key("llama32", 8192, "vt", "quest_streaming", 512, 160, 8, 16, 2)
            == cell_key("llama32", 8192, "vt", "quest_streaming", 512, 999, 99, 16, 2))
    assert (cell_key("llama32", 8192, "vt", "shadowkv", 512, 160, 8, 16, 2)
            == cell_key("llama32", 8192, "vt", "shadowkv", 512, 160, 8, 99, 9))
    # full reads none of them
    assert (cell_key("llama32", 8192, "vt", "full", 0, 160, 8, 16, 2)
            == cell_key("llama32", 8192, "vt", "full", 7, 1, 2, 3, 4))


def test_every_knob_a_method_reads_is_in_its_name():
    def q(**kw):
        args = dict(model_key="llama32", datalen=8192, task="vt", method="quest_streaming",
                    budget=512, rank=160, chunk=8, page=16, dense=2, group_reduce="max")
        args.update(kw)
        return cell_key(**args)

    assert q() != q(budget=1024)
    assert q() != q(page=32)
    assert q() != q(dense=0)
    assert q() != q(group_reduce="sum")

    def s(**kw):
        args = dict(model_key="llama32", datalen=8192, task="vt", method="shadowkv",
                    budget=512, rank=160, chunk=8, page=16, dense=2)
        args.update(kw)
        return cell_key(**args)

    assert s() != s(budget=1024)
    assert s() != s(rank=320)
    assert s() != s(chunk=16)


def test_m51f_names_the_same_cell_from_both_callers():
    """run_pool.sh claims a cell by name with only the nine queue fields;
    run_cell.sh writes it after passing the variant explicitly. If those two
    disagree the cell is claimed twice and two models land on one GPU -- which
    is exactly how five cells died on 2026-09-01."""
    line = ("qwen3", 8192, "vt", "m51f", 512, 8, "0.999", 8, 0)
    pooled = cell_key(*line)                       # what run_pool.sh computes
    explicit = cell_key(*line, "max", "a0softmax")  # what run_cell.sh passes
    assert pooled == explicit == "qwen3_8192_vt_m51f_b512_l8_a8_q0.999_d0_a0softmax"


def test_m51f_variant_tracks_the_environment(monkeypatch):
    monkeypatch.setenv("SHADOWKV_M51F_ANCHORS_RANKED", "2")
    monkeypatch.setenv("SHADOWKV_M51F_REDUCE", "max")
    assert cell_key("qwen3", 8192, "vt", "m51f", 512, 8, "0.999", 8, 0).endswith("_a2max")


def test_a_seeded_reasoning_task_names_a_distinct_cell():
    """Four seeds of one AIME configuration are four draws that must be kept
    apart. The seed cannot be an environment variable -- the pool's environment
    is fixed for a whole queue -- so it rides in the task key, and the task key
    is verbatim in the cell name. If it were not, the four cells would share one
    jsonl and each would overwrite the last."""
    names = {cell_key("qwen3", 32768, f"aime25-s{seed}", "quest_streaming",
                      1024, 160, 8, 8, 0) for seed in range(4)}
    assert len(names) == 4
    assert "aime25-s2" in cell_key("qwen3", 32768, "aime25-s2", "full",
                                   1024, 160, 8, 8, 0)


def test_the_generation_cap_is_in_the_cell_name():
    """A generation cap changes the number as surely as the budget does: at
    16384 a long chain of thought is cut off and scores wrong, at 32000 it
    finishes. Two runs of GPQA under different caps are two different
    measurements and must not share a jsonl."""
    at16k = cell_key("qwen3", 32768, "gpqa-s0", "pariskv_author_common",
                     1024, 160, 8, 8, 0)
    at32k = cell_key("qwen3", 32768, "gpqa-g32000-s0", "pariskv_author_common",
                     1024, 160, 8, 8, 0)
    assert at16k != at32k
    assert "gpqa-g32000-s0" in at32k
