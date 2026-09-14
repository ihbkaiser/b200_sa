#!/usr/bin/env python
"""The single definition of a cell's name.

Two different configurations must never produce the same key, and a key must
carry back everything needed to look the cell up by exact match -- so every
knob that changes the number is in the name, and knobs a method ignores are
left out rather than baked in as noise.
"""
import os
import sys

from outlier_policy import shadow_outlier_chunks


def m51c_codes():
    """PQ codewords for the m51c method. Same reason as m51f_variant: run_pool
    and run_cell both need this in the name and only one of them gets to pass
    arguments."""
    return os.environ.get("SHADOWKV_M51C_CODES", "256")


def m51f_variant():
    """The m51f implementation variant, derived from the environment.

    run_pool.sh names a cell to claim it and run_cell.sh names the same cell to
    write it, and they must agree -- a cell with two names is a cell that gets
    claimed twice and run twice on one GPU. run_pool.sh only knows the nine
    queue fields, so the variant cannot be an argument it has to remember to
    pass: it is derived here, from the same environment both scripts see.
    """
    return f"a{os.environ.get('SHADOWKV_M51F_ANCHORS_RANKED', '0')}" \
           f"{os.environ.get('SHADOWKV_M51F_REDUCE', 'softmax')}"


def cell_key(model_key, datalen, task, method, budget, rank, chunk, page, dense,
             group_reduce="max", variant=""):
    method = method.lower()
    if method == "full":
        tail = "full"
    elif method == "m51":
        # for m51 the "budget" slot carries the coverage target, not a token count,
        # and `variant` names the implementation modes -- pq_mode, group_select and
        # decode_mode all move the numbers, so they belong in the name
        tail = f"m51_c{budget}_l{page}_a{rank}_q{chunk}_d{dense}"
        if variant:
            tail += f"_{variant}"
    elif method == "m51f":
        # the fixed-budget M51: the budget slot is a TOKEN COUNT again, and the
        # traffic equals it, so it must not share a name shape with the
        # coverage-parameterised m51 above
        tail = f"m51f_b{budget}_l{page}_a{rank}_q{chunk}_d{dense}"
        tail += f"_{variant or m51f_variant()}"
    elif method == "m51w":
        # m51f with a warm-started PQ codebook: the offline codebook seeded into
        # k-means and refined for a couple of Lloyd steps on this context's
        # tangents. A separate method name rather than a flag, so the existing
        # m51f cells keep their names and nothing gets re-run.
        tail = f"m51w_b{budget}_l{page}_a{rank}_q{chunk}_d{dense}"
        tail += f"_{variant or m51f_variant()}"
    elif method == "m51c":
        # m51w at higher PQ resolution. Quest's page bound is exact given
        # min/max; M51 quantises its tangent, and on retrieval among
        # near-identical keys that quantisation is what loses -- 6 bits per 8
        # dims cost 20 RULER points on niah_multikey_3. The codeword count is in
        # the name because it changes the number AND the metadata footprint.
        tail = f"m51c{m51c_codes()}_b{budget}_l{page}_a{rank}_q{chunk}_d{dense}"
        tail += f"_{variant or m51f_variant()}"
    elif method == "quest_streaming":
        prefix = int(os.environ.get("QUEST_PREFIX_TOKENS", "0"))
        recent = int(os.environ.get("STREAMING_RECENT_TOKENS", "32"))
        tail = (
            f"quest_stream_b{budget}_p{page}_d{dense}_{group_reduce}"
            f"_x{prefix}_l{recent}"
        )
    elif method in {
        "exact_block_lse_streaming", "exact_block_max_streaming",
        "exact_block_lse_softmax_streaming",
        "exact_block_max_softmax_streaming",
    }:
        prefix = int(os.environ.get("QUEST_PREFIX_TOKENS", "32"))
        recent = int(os.environ.get("STREAMING_RECENT_TOKENS", "32"))
        statistic = "lse" if "_lse_" in method else "max"
        normalization = "_softmax" if "softmax" in method else ""
        query_mean = os.environ.get(
            "SHADOWKV_EXACT_QUERY_GROUP_MEAN", "0"
        ) == "1"
        gqa_mode = "qmean" if query_mean else group_reduce
        tail = (
            f"exact_block_{statistic}{normalization}_stream_b{budget}_c{chunk}"
            f"_{gqa_mode}_x{prefix}_l{recent}_oracle"
        )
        refine_factor = float(os.environ.get("STREAMING_REFINE_FACTOR", "1"))
        refine_tokens = os.environ.get("STREAMING_REFINE_TOKENS", "0") == "1"
        if refine_tokens:
            tail += f"_tokref{refine_factor:g}x"
    elif method == "pariskv_official":
        prefix = int(os.environ.get("QUEST_PREFIX_TOKENS", "32"))
        recent = int(os.environ.get("STREAMING_RECENT_TOKENS", "32"))
        tail = (
            f"pariskv_official_b{budget}_x{prefix}_l{recent}_polarann"
        )
    elif method == "pariskv_author_common":
        prefix = int(os.environ.get("QUEST_PREFIX_TOKENS", "32"))
        recent = int(os.environ.get("STREAMING_RECENT_TOKENS", "32"))
        tail = (
            f"pariskv_author_common_b{budget}_c{chunk}"
            f"_x{prefix}_l{recent}_polarann"
        )
    elif method == "retroinfer_author_common":
        prefix = int(os.environ.get("QUEST_PREFIX_TOKENS", "32"))
        recent = int(os.environ.get("STREAMING_RECENT_TOKENS", "32"))
        average = int(os.environ.get("RETROINFER_AVERAGE_CLUSTER_SIZE", "16"))
        centroids = int(os.environ.get("RETROINFER_N_CENTROIDS", "0"))
        segments = int(os.environ.get("RETROINFER_N_SEGMENT", "16"))
        estimation = os.environ.get("RETROINFER_ESTIMATION_RATIO", "0.232")
        iters = int(os.environ.get("RETROINFER_KMEANS_ITERS", "10"))
        tail = (
            f"retroinfer_author_common_b{budget}_c{chunk}_a{average}"
            f"_n{centroids}_g{segments}_e{estimation}_i{iters}"
            f"_x{prefix}_l{recent}_ivf"
        )
    elif method == "retroinfer_reference_streaming":
        prefix = int(os.environ.get("RETROINFER_PREFIX_TOKENS", "4"))
        recent = int(os.environ.get("RETROINFER_RECENT_TOKENS", "64"))
        update = int(os.environ.get("RETROINFER_UPDATE_SEGMENT", "1024"))
        average = int(os.environ.get("RETROINFER_AVERAGE_CLUSTER_SIZE", "16"))
        estimation = os.environ.get("RETROINFER_ESTIMATION_RATIO", "0.232")
        iters = int(os.environ.get("RETROINFER_KMEANS_ITERS", "10"))
        tail = (
            f"retroinfer_ref_stream_b{budget}_u{update}_a{average}"
            f"_e{estimation}_i{iters}_x{prefix}_l{recent}_exactkv"
        )
    elif method == "infllm_author_common":
        matched = os.environ.get("UPSTREAM_MATCHED_EXACT_REGIONS", "0") == "1"
        if matched:
            prefix = os.environ.get("QUEST_PREFIX_TOKENS", "32")
            recent = os.environ.get("STREAMING_RECENT_TOKENS", "32")
            tail = (
                f"infllm_author_common_routeb{budget}_x{prefix}_l{recent}"
                "_blk128_repr4_matched"
            )
        else:
            # Author-structure budget stress: scale the native 2:1
            # local/global split to the requested total active-token cap.
            tail = f"infllm_author_common_b{budget}_blk128_repr4_native21"
    elif method == "magicpig_author_common":
        k_bits = os.environ.get("MAGICPIG_K", "10")
        tables = os.environ.get("MAGICPIG_L", "210")
        sink = os.environ.get("MAGICPIG_SINK_TOKENS", "4")
        local = os.environ.get("MAGICPIG_LOCAL_TOKENS", "64")
        seed = os.environ.get("MAGICPIG_SEED", "43")
        matched = os.environ.get("UPSTREAM_MATCHED_EXACT_REGIONS", "0") == "1"
        if matched:
            sink = os.environ.get("QUEST_PREFIX_TOKENS", "32")
            local = os.environ.get("STREAMING_RECENT_TOKENS", "32")
        tail = (
            f"magicpig_author_common_targetb{budget}_k{k_bits}l{tables}"
            f"_x{sink}_l{local}_d0_s{seed}"
            f"{'_matched' if matched else ''}"
        )
    elif method == "pqcache_author_common":
        subvec = os.environ.get("PQCACHE_SUBVECTORS", "2")
        bits = os.environ.get("PQCACHE_SUBBITS", "6")
        iters = os.environ.get("PQCACHE_KMEANS_ITERS", "10")
        sink = os.environ.get("PQCACHE_SINK_TOKENS", "32")
        recent = os.environ.get("PQCACHE_RECENT_RATIO", "0.5")
        seed = os.environ.get("PQCACHE_SEED", "4321")
        matched = os.environ.get("UPSTREAM_MATCHED_EXACT_REGIONS", "0") == "1"
        if matched:
            prefix = os.environ.get("QUEST_PREFIX_TOKENS", "32")
            local = os.environ.get("STREAMING_RECENT_TOKENS", "32")
            tail = (
                f"pqcache_author_common_routeb{budget}_pq{subvec}x{bits}"
                f"_i{iters}_x{prefix}_l{local}_s{seed}_exactkv_matched"
            )
        else:
            tail = (
                f"pqcache_author_common_b{budget}_pq{subvec}x{bits}"
                f"_i{iters}_x{sink}_recent{recent}_s{seed}_exactkv"
            )
    elif method in ("shadowkv", "shadowkv_cpu"):
        outliers = shadow_outlier_chunks(int(datalen), int(chunk))
        tail = f"{method}_b{budget}_r{rank}_c{chunk}_o{outliers}"
    elif method == "shadowkv_quill":
        fraction = os.environ.get("SHADOWKV_QUILL_EXACT_FRACTION", "0.125")
        score_chunk = os.environ.get("SHADOWKV_QUILL_SCORE_CHUNK", "1024")
        outliers = shadow_outlier_chunks(int(datalen), int(chunk))
        tail = (
            f"shadowkv_quill_b{budget}_r{rank}_c{chunk}"
            f"_o{outliers}_ef{fraction}_sc{score_chunk}"
        )
    elif method in {"adaptive_centroid_lse", "adaptive_centroid_lse_prefix4"}:
        extra = os.environ.get("ADAPTIVE_LSE_EXTRA_FRACTION", "0.25")
        temps = os.environ.get("ADAPTIVE_LSE_TEMPERATURES", "1")
        prefix = "_prefix4" if method.endswith("_prefix4") else ""
        tail = (
            f"adaptive_lse{prefix}_b{budget}_c{chunk}"
            f"_x{extra}_t{temps.replace(',', '-')}_o0_exactkv"
        )
    elif method in {
        "adaptive_centroid_lse_streaming",
        "adaptive_centroid_lse_streaming_prefix4",
        "adaptive_centroid_lse_streaming_prefix4_querymean",
    }:
        querymean = method.endswith("_querymean")
        extra = os.environ.get(
            "ADAPTIVE_LSE_EXTRA_FRACTION", "0" if querymean else "0.25"
        )
        temps = os.environ.get("ADAPTIVE_LSE_TEMPERATURES", "1")
        recent = os.environ.get("STREAMING_RECENT_TOKENS", "32")
        explicit_prefix = os.environ.get("QUEST_PREFIX_TOKENS")
        prefix = (
            f"_p{explicit_prefix}" if explicit_prefix is not None
            else "_prefix4" if "_prefix4" in method else ""
        )
        reduce = "qmean" if querymean else group_reduce
        placement = os.environ.get(
            "SHADOWKV_CENTER_PLACEMENT",
            "self",
        ).strip().lower()
        placement_tag = "" if placement == "self" else f"_place{placement}"
        if placement == "qmass_bank":
            bank = os.environ.get("SHADOWKV_QUERY_MASS_BANK_SIZE", "32")
            window = os.environ.get("SHADOWKV_QUERY_MASS_WINDOW", "512")
            placement_tag += f"{bank}w{window}"
        if placement in {"qmass_one", "qmass_bank"}:
            objective = os.environ.get(
                "SHADOWKV_QUERY_MASS_OBJECTIVE", "global_mass"
            ).strip().lower()
            if objective != "global_mass":
                placement_tag += f"_{objective}"
            qmass_gqa = os.environ.get(
                "SHADOWKV_QUERY_MASS_GQA_REDUCE", "mean"
            ).strip().lower()
            if qmass_gqa != "mean":
                placement_tag += f"_gqa{qmass_gqa}"
        if (
            placement == "robust_trimmed"
            and os.environ.get("SHADOWKV_ROBUST_QUERY_SOURCE", "self_k")
            == "final"
        ):
            placement_tag += "_proxyqfinal"
        allocation = os.environ.get(
            "SHADOWKV_CENTER_ALLOCATION",
            "tail_absolute_rate_distortion" if querymean else "self_lse",
        ).strip().lower()
        allocation_tag = (
            "" if allocation == "self_lse" else f"_alloc{allocation}"
        )
        if placement == "robust_trimmed":
            trim = os.environ.get("SHADOWKV_ROBUST_TRIM_FRACTION", "0.25")
            allocation_tag += f"_trim{trim}"
        if allocation in {
            "tail_cvar", "anchor_tail_cvar", "density_tail_cvar",
            "tail_rate_distortion", "tail_relative_rate_distortion",
            "tail_absolute_rate_distortion", "tail_absolute_marginal",
            "tail_mass_rate_distortion",
            "tail_distortion_threshold", "tail_group_distortion_target",
            "tail_demand_adaptive_rate_distortion",
            "tail_hierarchical_log_rate_distortion",
            "tail_hierarchical_floor_rate_distortion",
        }:
            tail = os.environ.get("SHADOWKV_TAIL_CVAR_FRACTION", "0.25")
            allocation_tag += f"_tail{tail}"
            gap_scale = os.environ.get(
                "SHADOWKV_TAIL_GAP_CORRECTION_SCALE",
                "0",
            )
            if float(gap_scale) != 0.0:
                allocation_tag += f"_gapcorr{gap_scale}"
        if allocation in {
            "tail_absolute_rate_distortion", "tail_absolute_marginal",
            "tail_mass_rate_distortion",
        }:
            penalty = os.environ.get(
                "SHADOWKV_ABSOLUTE_RD_PENALTY",
                "1.5" if querymean else "0.25",
            )
            allocation_tag += f"_ard{penalty}"
        if allocation == "tail_distortion_threshold":
            tolerance = os.environ.get("SHADOWKV_DISTORTION_TOLERANCE", "1.0")
            allocation_tag += f"_tol{tolerance}"
        if allocation == "tail_group_distortion_target":
            target = os.environ.get("SHADOWKV_GROUP_DISTORTION_TARGET", "1.0")
            allocation_tag += f"_gdt{target}"
        if allocation == "tail_demand_adaptive_rate_distortion":
            low = os.environ.get("SHADOWKV_DEMAND_LOW_PENALTY", "1.5")
            high = os.environ.get("SHADOWKV_DEMAND_HIGH_PENALTY", "2.5")
            demand = os.environ.get("SHADOWKV_DEMAND_THRESHOLD", "1.75")
            allocation_tag += f"_darl{low}h{high}t{demand}"
        if allocation == "tail_hierarchical_log_rate_distortion":
            alpha = os.environ.get("SHADOWKV_HIERARCHICAL_RD_ALPHA", "3.0")
            beta = os.environ.get("SHADOWKV_HIERARCHICAL_RD_BETA", "1.0")
            group = os.environ.get("SHADOWKV_HIERARCHICAL_RD_GROUP", "head")
            allocation_tag += f"_hlrda{alpha}b{beta}g{group}"
        if allocation == "tail_hierarchical_floor_rate_distortion":
            base = os.environ.get("SHADOWKV_HIERARCHICAL_RD_BASE", "1.5")
            alpha = os.environ.get("SHADOWKV_HIERARCHICAL_RD_ALPHA", "1.5")
            beta = os.environ.get("SHADOWKV_HIERARCHICAL_RD_BETA", "1.0")
            group = os.environ.get("SHADOWKV_HIERARCHICAL_RD_GROUP", "head")
            allocation_tag += f"_hfrdl{base}a{alpha}b{beta}g{group}"
        if allocation == "density_tail_cvar":
            page = os.environ.get("SHADOWKV_DENSITY_PAGE_TOKENS", "1024")
            refs = os.environ.get("SHADOWKV_DENSITY_REFERENCE_COUNT", "64")
            kappa = os.environ.get("SHADOWKV_DENSITY_KAPPA", "8")
            shrink = os.environ.get("SHADOWKV_DENSITY_SHRINKAGE", "0.5")
            signal_mode = os.environ.get(
                "SHADOWKV_DENSITY_SIGNAL_MODE", "contrast"
            )
            weight_mode = os.environ.get(
                "SHADOWKV_DENSITY_WEIGHT_MODE", "symmetric"
            )
            head_trust = os.environ.get(
                "SHADOWKV_DENSITY_HEAD_TRUST_POWER", "0"
            )
            value_power = os.environ.get(
                "SHADOWKV_DENSITY_VALUE_POWER", "0"
            )
            allocation_tag += (
                f"_dpage{page}_dref{refs}_dk{kappa}_dshrink{shrink}"
            )
            if float(head_trust) != 0.0:
                allocation_tag += f"_dht{head_trust}"
            if float(value_power) != 0.0:
                allocation_tag += f"_dvp{value_power}"
            if signal_mode != "contrast":
                allocation_tag += f"_dsig{signal_mode}"
            if weight_mode != "symmetric":
                allocation_tag += f"_dweight{weight_mode}"
        if allocation == "exposure_angular_marginal":
            exposure_refs = os.environ.get(
                "SHADOWKV_EXPOSURE_REFERENCE_COUNT", "64"
            )
            allocation_tag += f"_eref{exposure_refs}"
        cost = os.environ.get(
            "SHADOWKV_SELF_LSE_COST", "mean_gap" if querymean else "max_gap"
        ).strip().lower()
        cost_tag = ""
        if cost != "max_gap":
            if cost in {"mean_gap", "mean_relative"}:
                cost_tag = f"_cost{cost}"
            else:
                beta = os.environ.get("SHADOWKV_SELF_LSE_COST_BETA", "4")
                cost_tag = f"_cost{cost}_beta{beta}"
        correction_tag = (
            "" if os.environ.get(
                "SHADOWKV_CENTER_DISPERSION_CORRECTION",
                "0" if querymean else "1",
            ) == "1" else "_noalpha"
        )
        refine = os.environ.get("STREAMING_REFINE_FACTOR", "1")
        refine_tag = "" if float(refine) == 1.0 else f"_rf{refine}"
        refine_ratio = os.environ.get("STREAMING_REFINE_CANDIDATE_RATIO")
        if refine_ratio is not None:
            refine_tag += f"_rr{refine_ratio}"
        if os.environ.get("STREAMING_REFINE_TOKENS", "0") == "1":
            refine_tag += "_rtok"
        sketch_rank = int(os.environ.get("STREAMING_REFINE_SKETCH_RANK", "0"))
        if sketch_rank:
            sketch_bits = int(os.environ.get(
                "STREAMING_REFINE_SKETCH_BITS", "4"
            ))
            refine_tag += f"_sketchr{sketch_rank}b{sketch_bits}"
            sketch_basis = os.environ.get(
                "STREAMING_REFINE_SKETCH_BASIS", "query_aware"
            )
            if sketch_basis != "query_aware":
                refine_tag += f"_{sketch_basis}"
        if os.environ.get(
            "STREAMING_REFINE_COMPONENT_CANDIDATES", "0"
        ) == "1":
            refine_tag += "_ccomp"
        # Variable-order 1..S is now part of the method identity.  Including
        # it unconditionally prevents reuse of legacy one/two-center cells.
        max_components = os.environ.get("STREAMING_MAX_COMPONENTS", str(chunk))
        refine_tag += f"_rmax{max_components}"
        compact = os.environ.get(
            "STREAMING_COMPACT_METADATA", "1" if querymean else "0"
        ) == "1"
        center_bits = int(os.environ.get(
            "STREAMING_CENTER_BITS", "8" if querymean else "16"
        ))
        metadata_tag = ("_compact" if compact else "") + (
            "" if center_bits == 16 else f"_cb{center_bits}"
        )
        # The two router backends are deterministic but not equal to each
        # other: at 32K b1024 they score 0.780 and 0.785 on cwe from different
        # predictions.  A knob that changes the number belongs in the name, and
        # naming BOTH of them keeps cells written before this distinction
        # existed from being mistaken for either.
        router = os.environ.get("STREAMING_ROUTER_BACKEND", "triton")
        tail = (
            f"adaptive_lse_stream{prefix}_b{budget}_c{chunk}"
            f"_x{extra}_t{temps.replace(',', '-')}_{reduce}"
            f"_l{recent}{placement_tag}{allocation_tag}"
            f"{cost_tag}{correction_tag}{refine_tag}{metadata_tag}"
            f"_rb{router}_exactkv"
        )
    else:
        raise SystemExit(f"unknown method '{method}'")
    return f"{model_key}_{datalen}_{task}_{tail}"


if __name__ == "__main__":
    if len(sys.argv) not in (10, 11, 12):
        raise SystemExit("usage: cell_key.py model_key datalen task method budget rank "
                         "chunk page dense [group_reduce|variant]")
    print(cell_key(*sys.argv[1:]))
