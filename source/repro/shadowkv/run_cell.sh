#!/usr/bin/env bash
# Run exactly one cell. This is what a pool worker calls, and it is also the
# right way to run a cell by hand -- there is no second code path.
#
#   repro/shadowkv/run_cell.sh <model_key> <datalen> <task> <method> \
#                              <budget> <rank> <chunk> <page> <dense> <num_samples> <gpu>
#
# model_key: llama32 | qwen3      (resolved to a path by env_m1.sh)
# method   : full | quest_streaming | exact_block_lse_streaming |
#            exact_block_max_streaming | exact_block_{lse,max}_softmax_streaming |
#            pariskv_official |
#            infllm_author_common |
#            magicpig_author_common |
#            pqcache_author_common |
#            retroinfer_reference_streaming | retroinfer_author_common |
#            shadowkv | shadowkv_cpu | adaptive_centroid_lse_streaming[_prefix4]
set -euo pipefail

: "${SHADOWKV_DIR:?source repro/shadowkv/env_m1.sh first}"
: "${PY:?source repro/shadowkv/env_m1.sh first}"
: "${SHADOWKV_RESULTS_ROOT:?source repro/shadowkv/env_m1.sh first}"

# Head-allocation modes are inert unless the ragged allocator is enabled.
# Silently accepting MODE/regularizer without TAU previously produced uniform
# controls mislabeled as adaptive experiments.  Fail before loading a model so
# this protocol error cannot consume GPU time or contaminate result roots.
if [ -n "${SHADOWKV_HEAD_ALLOC_MODE:-}" ] && [ -z "${SHADOWKV_HEAD_ALLOC_TAU:-}" ]; then
  echo "ERROR: SHADOWKV_HEAD_ALLOC_MODE requires SHADOWKV_HEAD_ALLOC_TAU" >&2
  exit 2
fi
if [ "${SHADOWKV_HEAD_ALLOC_REGULARIZER:-0}" != 0 ] && [ -z "${SHADOWKV_HEAD_ALLOC_TAU:-}" ]; then
  echo "ERROR: head-allocation regularizer requires SHADOWKV_HEAD_ALLOC_TAU" >&2
  exit 2
fi

MODEL_KEY=$1; DATALEN=$2; TASK=$3; METHOD=$4
BUDGET=$5; RANK=$6; CHUNK=$7; PAGE=$8; DENSE=$9; NUM_SAMPLES=${10}; GPU=${11}
GROUP_REDUCE=${SHADOWKV_GROUP_REDUCE_OVERRIDE:-${12:-max}}

# Residency is a deployment choice, not an algorithm, so it follows one rule
# for every method (PROJECT.md 2e).  That rule is now "always offload", and the
# reason is not memory but speed: the fused UVA gather and its cross-step reuse
# only run on the offloaded path (_can_reuse_selected_blocks requires it), so a
# GPU-resident cache falls back to the generic gather and loses the reuse.
# Measured at 32K b1024, ours: offloaded prefill 15.05 s / decode median
# 43.11 ms / peak 10.91 GiB, GPU-resident 27.48 s / 49.91 ms / 15.47 GiB -- and
# the resident run even started on a cooler card.  At 64K resident also simply
# OOMs for ours, Quest and ParisKV on a 24 GB board.
# Set STREAMING_OFFLOAD=0 explicitly to measure the resident path on purpose.
export STREAMING_OFFLOAD="${STREAMING_OFFLOAD:-1}"

# At 128K ParisKV fails on a 24 GB card with 3.13 GiB reserved-but-unallocated
# -- fragmentation, not capacity.  The expandable allocator recovers it and the
# cell then runs.  It is an allocator setting, not a numerical one: verified
# byte-identical predictions on quest_streaming cwe 32K b1024, 8 samples.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Queue task keys are filesystem-safe. Resolve them once to the Dataset adapter
# name; legacy bare task names continue to mean RULER.
case "$TASK" in
  longbench-v2) DATASET_NAME=$TASK ;;
  longbench-v2-short)
    DATASET_NAME=longbench-v2
    export SHADOWKV_LONGBENCH_V2_LENGTH_FILTER=short ;;
  longbench-v2-medium)
    DATASET_NAME=longbench-v2
    export SHADOWKV_LONGBENCH_V2_LENGTH_FILTER=medium ;;
  # A reasoning task key carries the knobs that vary cell by cell:
  #
  #     <bench>[-g<max_new_tokens>][-s<seed>]     e.g. gpqa-g32000-s2
  #
  # The pool's environment is fixed for a whole queue while the nine queue
  # fields are not, so anything that changes per cell has to travel in a field.
  # It rides in the task, the way LongBench-v2's length band does, and that
  # also puts it in the cell name for free -- which is the point: two seeds, or
  # two generation caps, that share a name are two writes to one jsonl, and a
  # generation cap changes the number as surely as the budget does.
  aime25*|math500*|gpqa*)
    _rt=$TASK
    case "$_rt" in *-s*) export SHADOWKV_GENERATION_SEED=${_rt##*-s}; _rt=${_rt%-s*} ;; esac
    case "$_rt" in *-g*) export SHADOWKV_MAX_NEW_TOKENS=${_rt##*-g}; _rt=${_rt%-g*} ;; esac
    case "$_rt" in
      gpqa|gpqa-diamond) DATASET_NAME=gpqa/diamond ;;
      *) DATASET_NAME=$_rt ;;
    esac
    unset _rt ;;
  longbench-*) DATASET_NAME="longbench/${TASK#longbench-}" ;;
  *) DATASET_NAME="ruler/$TASK" ;;
esac
# The m51 implementation variant is campaign-level, not per cell, so it comes
# from the environment: the pool derives the cell name from the same variable,
# and a queue line stays exactly 9 fields.
#   v2 = offline codebook / per-query order / dense-mask decode  (default)
#   v1 = per-context codebook / per-query order / dense mask     (probe-faithful)
M51_VARIANT=${SHADOWKV_M51_VARIANT:-v2}
OUTLIER_CHUNKS=$("$PY" "$(dirname "${BASH_SOURCE[0]}")/outlier_policy.py" "$DATALEN" "$CHUNK")

case "$MODEL_KEY" in
  llama32) MODEL_PATH=$SHADOWKV_LLAMA32_PATH; TEMPLATE=llama-3
           M51_ANCHORS=${SHADOWKV_M51_ANCHORS_LLAMA32:-} ;;
  qwen3)   MODEL_PATH=$SHADOWKV_QWEN3_PATH;   TEMPLATE=qwen
           M51_ANCHORS=${SHADOWKV_M51_ANCHORS_QWEN3:-} ;;
  *) echo "unknown model_key '$MODEL_KEY'"; exit 1 ;;
esac
[ -n "$MODEL_PATH" ] || { echo "empty model path for '$MODEL_KEY'"; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "model path not found: $MODEL_PATH"; exit 1; }

CELL=$("$PY" "$(dirname "${BASH_SOURCE[0]}")/cell_key.py" \
        "$MODEL_KEY" "$DATALEN" "$TASK" "$METHOD" "$BUDGET" "$RANK" "$CHUNK" "$PAGE" "$DENSE" \
        "$([ "$METHOD" = m51 ] || [ "$METHOD" = m51f ] && echo max || echo "$GROUP_REDUCE")" \
        "$([ "$METHOD" = m51 ] && echo "$M51_VARIANT" \
           || { [ "$METHOD" = m51f ] && echo "a${SHADOWKV_M51F_ANCHORS_RANKED:-0}${SHADOWKV_M51F_REDUCE:-softmax}"; } \
           || echo "")")
[ -n "$CELL" ] || { echo "empty cell key -- refusing to run"; exit 1; }

OUT_ROOT=$SHADOWKV_RESULTS_ROOT/cells
LOG_DIR=$SHADOWKV_RESULTS_ROOT/_logs
mkdir -p "$OUT_ROOT" "$LOG_DIR"    # before any redirect touches them

EXTRA=()
# the queue's method name and the flag eval_acc takes are not always the same:
# m51f is a cell-naming distinction over the same --method m51 code path
METHOD_ARG=$METHOD
case "$METHOD" in
  full)     ;;
  quest_streaming)
            EXTRA=(--sparse_budget "$BUDGET" --page_size "$PAGE" --dense_layers "$DENSE" --group_reduce "$GROUP_REDUCE"
                   --quest_prefix_tokens "${QUEST_PREFIX_TOKENS:-0}"
                   --streaming_recent_tokens "${STREAMING_RECENT_TOKENS:-32}" \
                   --streaming_update_interval "${STREAMING_UPDATE_INTERVAL:-0}")
            [ "${STREAMING_OFFLOAD:-0}" = 1 ] && EXTRA+=(
                --streaming_offload
                --streaming_gather_backend "${STREAMING_GATHER_BACKEND:-auto}"
            ) ;;
  exact_block_lse_streaming|exact_block_max_streaming|exact_block_lse_softmax_streaming|exact_block_max_softmax_streaming)
            EXTRA=(--sparse_budget "$BUDGET" --chunk_size "$CHUNK"
                   --group_reduce "$GROUP_REDUCE"
                   --quest_prefix_tokens "${QUEST_PREFIX_TOKENS:-32}"
                   --streaming_recent_tokens "${STREAMING_RECENT_TOKENS:-32}" \
                   --streaming_update_interval "${STREAMING_UPDATE_INTERVAL:-0}")
            EXTRA+=(--streaming_refine_factor "${STREAMING_REFINE_FACTOR:-1}")
            [ "${STREAMING_REFINE_TOKENS:-0}" != 1 ] || EXTRA+=(--streaming_refine_tokens)
            [ "${STREAMING_OFFLOAD:-0}" = 1 ] && EXTRA+=(--streaming_offload --streaming_gather_backend "${STREAMING_GATHER_BACKEND:-auto}")
            [ -z "${SHADOWKV_EXACT_ADAPTIVE_AUDIT:-}" ] || \
              EXTRA+=(--traffic_out "$SHADOWKV_EXACT_ADAPTIVE_AUDIT") ;;
  pariskv_official)
            : "${PARISKV_AUTHOR_ROOT:?set PARISKV_AUTHOR_ROOT to the official checkout}"
            [ "$MODEL_KEY" = qwen3 ] || {
              echo "ParisKV's official runtime currently supports the Qwen path only" >&2
              exit 2
            } ;;
  pariskv_author_common)
            : "${PARISKV_AUTHOR_ROOT:?set PARISKV_AUTHOR_ROOT to the official checkout}"
            EXTRA=(--sparse_budget "$BUDGET" --chunk_size "$CHUNK"
                   --pariskv_author_root "$PARISKV_AUTHOR_ROOT"
                   --quest_prefix_tokens "${QUEST_PREFIX_TOKENS:-32}"
                   --streaming_recent_tokens "${STREAMING_RECENT_TOKENS:-32}" \
                   --streaming_update_interval "${STREAMING_UPDATE_INTERVAL:-0}")
            [ "${STREAMING_OFFLOAD:-0}" = 1 ] && EXTRA+=(
                --streaming_offload
                --streaming_gather_backend "${STREAMING_GATHER_BACKEND:-auto}"
            ) ;;
  retroinfer_author_common)
            : "${RETROINFER_AUTHOR_ROOT:?set RETROINFER_AUTHOR_ROOT to the RetrievalAttention checkout}"
            EXTRA=(--sparse_budget "$BUDGET" --chunk_size "$CHUNK"
                   --retroinfer_author_root "$RETROINFER_AUTHOR_ROOT"
                   --retroinfer_average_cluster_size "${RETROINFER_AVERAGE_CLUSTER_SIZE:-16}"
                   --retroinfer_n_centroids "${RETROINFER_N_CENTROIDS:-0}"
                   --retroinfer_n_segment "${RETROINFER_N_SEGMENT:-16}"
                   --retroinfer_estimation_ratio "${RETROINFER_ESTIMATION_RATIO:-0.232}"
                   --retroinfer_kmeans_iters "${RETROINFER_KMEANS_ITERS:-10}"
                   --quest_prefix_tokens "${QUEST_PREFIX_TOKENS:-32}"
                   --streaming_recent_tokens "${STREAMING_RECENT_TOKENS:-32}" \
                   --streaming_update_interval "${STREAMING_UPDATE_INTERVAL:-0}")
            [ "${STREAMING_OFFLOAD:-0}" = 1 ] && EXTRA+=(
                --streaming_offload
                --streaming_gather_backend "${STREAMING_GATHER_BACKEND:-auto}"
            ) ;;
  retroinfer_reference_streaming)
            EXTRA=(--sparse_budget "$BUDGET"
                   # RetroInfer's three region knobs ARE the shared frame's sink,
                   # local window and flush cadence, so they fall back to the
                   # campaign-wide variables before the author defaults. Left at
                   # update_segment 1024 it attends up to 1088 exact tokens on top
                   # of its budget -- roughly double what the other methods get.
                   --retroinfer_prefix_tokens "${RETROINFER_PREFIX_TOKENS:-${QUEST_PREFIX_TOKENS:-4}}"
                   --retroinfer_recent_tokens "${RETROINFER_RECENT_TOKENS:-${STREAMING_RECENT_TOKENS:-64}}"
                   --retroinfer_update_segment "${RETROINFER_UPDATE_SEGMENT:-${STREAMING_UPDATE_INTERVAL:-1024}}"
                   --retroinfer_average_cluster_size "${RETROINFER_AVERAGE_CLUSTER_SIZE:-16}"
                   --retroinfer_estimation_ratio "${RETROINFER_ESTIMATION_RATIO:-0.232}"
                   --retroinfer_kmeans_iters "${RETROINFER_KMEANS_ITERS:-10}") ;;
  infllm_author_common)
            : "${INFLLM_AUTHOR_ROOT:?set INFLLM_AUTHOR_ROOT to the official checkout}"
            [ -f "$INFLLM_AUTHOR_ROOT/inf_llm/attention/context_manager.py" ] || {
              echo "invalid INFLLM_AUTHOR_ROOT: $INFLLM_AUTHOR_ROOT" >&2
              exit 2
            }
            EXTRA=(--sparse_budget "$BUDGET"
                   --traffic_out "$OUT_ROOT/$MODEL_KEY/$CELL.traffic.jsonl") ;;
  magicpig_author_common)
            : "${MAGICPIG_AUTHOR_ROOT:?set MAGICPIG_AUTHOR_ROOT to the official checkout}"
            [ -f "$MAGICPIG_AUTHOR_ROOT/models/attnserver.py" ] || {
              echo "invalid MAGICPIG_AUTHOR_ROOT: $MAGICPIG_AUTHOR_ROOT" >&2
              exit 2
            }
            EXTRA=(--sparse_budget "$BUDGET"
                   --traffic_out "$OUT_ROOT/$MODEL_KEY/$CELL.traffic.jsonl") ;;
  pqcache_author_common)
            : "${PQCACHE_AUTHOR_ROOT:?set PQCACHE_AUTHOR_ROOT to the official checkout}"
            [ -f "$PQCACHE_AUTHOR_ROOT/vq_method/retrieval_based/pq_search.py" ] || {
              echo "invalid PQCACHE_AUTHOR_ROOT: $PQCACHE_AUTHOR_ROOT" >&2
              exit 2
            }
            EXTRA=(--sparse_budget "$BUDGET"
                   --traffic_out "$OUT_ROOT/$MODEL_KEY/$CELL.traffic.jsonl")
            [ "${STREAMING_OFFLOAD:-0}" = 1 ] && EXTRA+=(--streaming_offload) ;;
  m51)      # for m51 the positional slots carry: budget=coverage, page=leaf,
            # rank=n_anchors, chunk=eta_quantile. The anchor bank is per model
            # and must exist -- an m51 cell without one is not runnable.
            [ -n "$M51_ANCHORS" ] || { echo "no M51 anchor bank for '$MODEL_KEY' (set SHADOWKV_M51_ANCHORS_${MODEL_KEY^^})"; exit 1; }
            [ -f "$M51_ANCHORS" ] || { echo "M51 anchor bank not found: $M51_ANCHORS"; exit 1; }
            case "$M51_VARIANT" in
              v2|max) PQ=offline;     SEL=per_query; DEC=dense_mask ;;   # measured best: see _bench/m51_ablation.jsonl
              v1)     PQ=per_context; SEL=per_query; DEC=dense_mask ;;
              *) echo "unknown m51 variant '$M51_VARIANT'"; exit 1 ;;
            esac
            EXTRA=(--m51_anchors "$M51_ANCHORS" --m51_coverage "$BUDGET" --m51_leaf "$PAGE" \
                   --m51_anchors_n "$RANK" --m51_eta_q "$CHUNK" --dense_layers "$DENSE" \
                   --m51_pq_mode "$PQ" --m51_group_select "$SEL" --m51_decode_mode "$DEC" \
                   --traffic_out "$SHADOWKV_RESULTS_ROOT/_traffic/${MODEL_KEY}.jsonl") ;;
  m51c|m51w|m51f)  # fixed-budget M51: one shared order per KV head, top-k tokens.
            # Positional slots go back to their normal meaning -- budget is a
            # token count -- because this variant does not take a coverage.
            [ -n "$M51_ANCHORS" ] || { echo "no M51 anchor bank for '$MODEL_KEY'"; exit 1; }
            [ -f "$M51_ANCHORS" ] || { echo "M51 anchor bank not found: $M51_ANCHORS"; exit 1; }
            METHOD_ARG=m51
            # m51w differs from m51f in one knob: the PQ codebook is warm-started
            # from the offline one instead of used as-is. Measured at 8K, two
            # Lloyd steps recover 88% of a full per-context refit
            # (_bench/m51_warm.jsonl) -- the refit itself costs 3.8x the prefill.
            PQM=offline; [ "$METHOD" = m51w ] && PQM=warm
            CODES=64
            if [ "$METHOD" = m51c ]; then
              PQM=warm
              CODES=${SHADOWKV_M51C_CODES:-256}
              # a warm start needs an offline codebook of the SAME width, so the
              # higher-resolution method carries its own bank
              M51_ANCHORS=${SHADOWKV_M51C_BANK:-$M51_ANCHORS}
              [ -f "$M51_ANCHORS" ] || { echo "no m51c bank: $M51_ANCHORS"; exit 1; }
            fi
            EXTRA=(--m51_anchors "$M51_ANCHORS" --sparse_budget "$BUDGET" \
                   --m51_select_mode fixed --m51_leaf "$PAGE" --m51_anchors_n "$RANK" \
                   --m51_eta_q "$CHUNK" --dense_layers "$DENSE" --m51_pq_mode "$PQM" \
                   --m51_pq_warm_iters "${SHADOWKV_M51W_ITERS:-2}" \
                   --m51_pq_codes "$CODES" \
                   --m51_rank_anchors "${SHADOWKV_M51F_ANCHORS_RANKED:-0}" \
                   --m51_rank_reduce "${SHADOWKV_M51F_REDUCE:-softmax}" \
                   --traffic_out "$SHADOWKV_RESULTS_ROOT/_traffic/${MODEL_KEY}.jsonl") ;;
  shadowkv|shadowkv_cpu)
            EXTRA=(--sparse_budget "$BUDGET" --rank "$RANK" --chunk_size "$CHUNK"
                   --shadow_outlier_chunks "$OUTLIER_CHUNKS") ;;
  shadowkv_quill)
            EXTRA=(--sparse_budget "$BUDGET" --rank "$RANK" --chunk_size "$CHUNK"
                   --shadow_outlier_chunks "$OUTLIER_CHUNKS"
                   --quill_router_exact_fraction "${SHADOWKV_QUILL_EXACT_FRACTION:-0.125}"
                   --quill_router_score_chunk "${SHADOWKV_QUILL_SCORE_CHUNK:-1024}") ;;
  adaptive_centroid_lse|adaptive_centroid_lse_prefix4)
            EXTRA=(--sparse_budget "$BUDGET" --rank "$RANK" --chunk_size "$CHUNK"
                   --router_centroids "$CHUNK"
                   --router_centroid_method self_lse_adaptive_iso
                   --router_split_fraction "${ADAPTIVE_LSE_EXTRA_FRACTION:-0.25}"
                   --router_self_lse_temperatures "${ADAPTIVE_LSE_TEMPERATURES:-1}") ;;
  adaptive_centroid_lse_streaming|adaptive_centroid_lse_streaming_prefix4|adaptive_centroid_lse_streaming_prefix4_querymean)
            if [[ "$METHOD" == *prefix4* ]]; then
              DEFAULT_ADAPTIVE_PREFIX=$((4 * CHUNK))
            else
              DEFAULT_ADAPTIVE_PREFIX=0
            fi
            if [[ "$METHOD" == *_querymean ]]; then
              # Fixed-price RD has no global mean-center quota.
              ADAPTIVE_EXTRA=${ADAPTIVE_LSE_EXTRA_FRACTION:-0}
              DEFAULT_CENTER_BITS=8
              DEFAULT_COMPACT_METADATA=1
            else
              ADAPTIVE_EXTRA=${ADAPTIVE_LSE_EXTRA_FRACTION:-0.25}
              DEFAULT_CENTER_BITS=16
              DEFAULT_COMPACT_METADATA=0
            fi
            EXTRA=(--sparse_budget "$BUDGET" --chunk_size "$CHUNK"
                   --group_reduce "$GROUP_REDUCE"
                   --router_centroids "$CHUNK"
                   --router_centroid_method self_lse_adaptive_iso
                   --router_split_fraction "$ADAPTIVE_EXTRA"
                   --router_self_lse_temperatures "${ADAPTIVE_LSE_TEMPERATURES:-1}"
                   --quest_prefix_tokens "${QUEST_PREFIX_TOKENS:-$DEFAULT_ADAPTIVE_PREFIX}"
                   --streaming_recent_tokens "${STREAMING_RECENT_TOKENS:-32}" \
                   --streaming_update_interval "${STREAMING_UPDATE_INTERVAL:-0}")
            # triton is the benchmark default: same method, 1.6x faster
            # decode (median 43.0 vs 70.5 ms at 64K B=4096), and the two agree
            # to within noise on accuracy (32K b1024, 20 samples: cwe 0.780 vs
            # 0.785, niah_multikey_3 1.000 vs 1.000).  They are NOT bit-equal,
            # so a table must not mix them -- set STREAMING_ROUTER_BACKEND=torch
            # to reproduce a pre-2026-09-13 number.
            EXTRA+=(--streaming_router_backend "${STREAMING_ROUTER_BACKEND:-triton}")
            EXTRA+=(--streaming_refine_factor "${STREAMING_REFINE_FACTOR:-1}")
            [ -z "${STREAMING_REFINE_CANDIDATE_RATIO:-}" ] || EXTRA+=(--streaming_refine_candidate_ratio "$STREAMING_REFINE_CANDIDATE_RATIO")
            [ "${STREAMING_REFINE_TOKENS:-0}" != 1 ] || EXTRA+=(--streaming_refine_tokens)
            # The method is variable-order 1..S.  Never silently fall back to
            # the obsolete one/two-center implementation.
            EXTRA+=(--streaming_max_components "${STREAMING_MAX_COMPONENTS:-$CHUNK}")
            EXTRA+=(--streaming_center_bits "${STREAMING_CENTER_BITS:-$DEFAULT_CENTER_BITS}")
            if [ "${STREAMING_COMPACT_METADATA:-$DEFAULT_COMPACT_METADATA}" = 1 ]; then
              EXTRA+=(--streaming_compact_metadata)
            else
              EXTRA+=(--no_streaming_compact_metadata)
            fi
            [ -z "${STREAMING_ALLOCATION_OUT:-}" ] || EXTRA+=(--allocation_out "$STREAMING_ALLOCATION_OUT")
            [ -z "${STREAMING_HEAD_ALLOCATION_OUT:-}" ] || EXTRA+=(--traffic_out "$STREAMING_HEAD_ALLOCATION_OUT")
            [ "${STREAMING_OFFLOAD:-0}" = 1 ] && EXTRA+=(--streaming_offload --streaming_gather_backend "${STREAMING_GATHER_BACKEND:-auto}") ;;
  *) echo "unknown method '$METHOD'"; exit 1 ;;
esac

# Runtime mode: measure prefill and per-step decode latency instead of scoring.
# It runs through the same eval_acc invocation so a timing cell cannot silently
# configure a method differently from the accuracy cell it is compared against.
if [ -n "${SHADOWKV_RUNTIME_OUT:-}" ]; then
  EXTRA+=(--runtime_out "$SHADOWKV_RUNTIME_OUT"
          --runtime_steps "${SHADOWKV_RUNTIME_STEPS:-64}"
          --runtime_warmup "${SHADOWKV_RUNTIME_WARMUP:-8}")
fi

cd "$SHADOWKV_DIR"   # data/ruler/data is resolved relative to the repo root

# stamp: what code and what packages produced this number
STAMP=$OUT_ROOT/$MODEL_KEY/$CELL.stamp.json
mkdir -p "$(dirname "$STAMP")"
"$PY" - "$STAMP" "$CELL" "$MODEL_PATH" "$METHOD" "$DATALEN" "$BUDGET" "$PAGE" "$DENSE" "$RANK" "$CHUNK" "$NUM_SAMPLES" "$GPU" "$OUTLIER_CHUNKS" "$GROUP_REDUCE" <<'PYEOF'
import json, subprocess, sys, os
import torch, transformers, flash_attn
(out, cell, model, method, datalen, budget, page, dense, rank, chunk, nsamples, gpu,
 outlier_chunks, group_reduce) = sys.argv[1:15]
repo = os.environ.get("CODE", ".")
def git(*a):
    try: return subprocess.check_output(["git", "-C", repo, *a], text=True).strip()
    except Exception: return None
def author_stamp(env_name):
    root = os.environ.get(env_name)
    if not root:
        return None
    try:
        sha = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        sha = None
    return {"root": root, "commit": sha}
def num(x):
    # the positional slots carry different types per method: m51 puts a
    # coverage target (0.90) and an eta quantile (0.999) where the sparse
    # methods put token counts. Parse leniently instead of assuming ints.
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x

querymean_default = method.endswith("_querymean")
center_placement = os.environ.get(
    "SHADOWKV_CENTER_PLACEMENT",
    "self",
).strip().lower()
center_allocation = os.environ.get(
    "SHADOWKV_CENTER_ALLOCATION",
    "tail_absolute_rate_distortion" if querymean_default else "self_lse",
).strip().lower()
default_gap_correction = "0"
default_self_lse_cost = "mean_gap" if querymean_default else "max_gap"
default_absolute_rd_penalty = "1.5" if querymean_default else "0.25"

json.dump({
    "cell": cell, "model_path": model, "method": method,
    "datalen": int(datalen), "sparse_budget": num(budget),
    "page_size": num(page), "dense_layers": num(dense),
    "rank": num(rank), "chunk_size": num(chunk), "num_samples": int(nsamples),
    "max_new_tokens_override": (
        int(os.environ["SHADOWKV_MAX_NEW_TOKENS"])
        if os.environ.get("SHADOWKV_MAX_NEW_TOKENS") else None
    ),
    "generation_temperature": float(
        os.environ.get("SHADOWKV_GENERATION_TEMPERATURE", "0")
    ),
    "generation_top_p": float(
        os.environ.get("SHADOWKV_GENERATION_TOP_P", "1")
    ),
    "generation_top_k": int(
        os.environ.get("SHADOWKV_GENERATION_TOP_K", "-1")
    ),
    "generation_seed": (
        int(os.environ["SHADOWKV_GENERATION_SEED"])
        if os.environ.get("SHADOWKV_GENERATION_SEED") else None
    ),
    "longbench_v2_length_filter": os.environ.get(
        "SHADOWKV_LONGBENCH_V2_LENGTH_FILTER"
    ),
    "group_reduce": group_reduce,
    "center_placement": (
        center_placement
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "center_allocation": (
        center_allocation
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "query_mass_bank_size": (
        int(os.environ.get("SHADOWKV_QUERY_MASS_BANK_SIZE", "32"))
        if center_placement == "qmass_bank" else None
    ),
    "query_mass_objective": (
        os.environ.get("SHADOWKV_QUERY_MASS_OBJECTIVE", "global_mass")
        if center_placement in {"qmass_one", "qmass_bank"} else None
    ),
    "query_mass_gqa_reduce": (
        os.environ.get("SHADOWKV_QUERY_MASS_GQA_REDUCE", "mean")
        if center_placement in {"qmass_one", "qmass_bank"} else None
    ),
    "query_mass_window": (
        int(os.environ.get("SHADOWKV_QUERY_MASS_WINDOW", "512"))
        if center_placement == "qmass_bank" else None
    ),
    "shadow_outlier_chunks": (
        0 if method.startswith("adaptive_centroid_lse")
        else int(outlier_chunks) if method.startswith("shadowkv")
        else None
    ),
    "quest_prefix_tokens": (
        int(os.environ.get(
            "QUEST_PREFIX_TOKENS",
            "32" if method in {
                "pariskv_official", "pariskv_author_common",
                "retroinfer_author_common"
            } else "0",
        ))
        if method in {"quest_streaming", "exact_block_lse_streaming",
                      "exact_block_max_streaming", "exact_block_lse_softmax_streaming",
                      "exact_block_max_softmax_streaming", "pariskv_official",
                      "pariskv_author_common", "retroinfer_author_common"}
        else int(os.environ.get(
            "QUEST_PREFIX_TOKENS",
            str(4 * int(chunk)) if "_prefix4" in method else "0",
        )) if method.startswith("adaptive_centroid_lse_streaming")
        else None
    ),
    "quest_recent_tokens": (
        (int(os.environ.get("STREAMING_RECENT_TOKENS", "32"))
         if "streaming" in method or method in {
             "pariskv_official", "pariskv_author_common",
             "retroinfer_author_common"
         } else None)
    ),
    "pariskv_author_root": (
        os.environ.get("PARISKV_AUTHOR_ROOT")
        if method in {"pariskv_official", "pariskv_author_common"} else None
    ),
    "pqcache_author": (
        author_stamp("PQCACHE_AUTHOR_ROOT")
        if method == "pqcache_author_common" else None
    ),
    "magicpig_author": (
        author_stamp("MAGICPIG_AUTHOR_ROOT")
        if method == "magicpig_author_common" else None
    ),
    "infllm_author": (
        author_stamp("INFLLM_AUTHOR_ROOT")
        if method == "infllm_author_common" else None
    ),
    "magicpig_k_l": (
        [int(os.environ.get("MAGICPIG_K", "10")),
         int(os.environ.get("MAGICPIG_L", "210"))]
        if method == "magicpig_author_common" else None
    ),
    "magicpig_seed": (
        int(os.environ.get("MAGICPIG_SEED", "43"))
        if method == "magicpig_author_common" else None
    ),
    "pqcache_pq": (
        [int(os.environ.get("PQCACHE_SUBVECTORS", "2")),
         int(os.environ.get("PQCACHE_SUBBITS", "6"))]
        if method == "pqcache_author_common" else None
    ),
    "pqcache_seed": (
        int(os.environ.get("PQCACHE_SEED", "4321"))
        if method == "pqcache_author_common" else None
    ),
    "upstream_matched_exact_regions": (
        os.environ.get("UPSTREAM_MATCHED_EXACT_REGIONS", "0") == "1"
        if method in {"pqcache_author_common", "magicpig_author_common",
                      "infllm_author_common"} else None
    ),
    "upstream_prefix_tokens": (
        int(os.environ.get("QUEST_PREFIX_TOKENS", "32"))
        if method in {"pqcache_author_common", "magicpig_author_common",
                      "infllm_author_common"}
        and os.environ.get("UPSTREAM_MATCHED_EXACT_REGIONS", "0") == "1"
        else None
    ),
    "upstream_recent_tokens": (
        int(os.environ.get("STREAMING_RECENT_TOKENS", "32"))
        if method in {"pqcache_author_common", "magicpig_author_common",
                      "infllm_author_common"}
        and os.environ.get("UPSTREAM_MATCHED_EXACT_REGIONS", "0") == "1"
        else None
    ),
    "retroinfer_author": (
        author_stamp("RETROINFER_AUTHOR_ROOT")
        if method == "retroinfer_author_common" else None
    ),
    "retroinfer_n_centroids": (
        int(os.environ.get("RETROINFER_N_CENTROIDS", "0"))
        if method == "retroinfer_author_common" else None
    ),
    "retroinfer_n_segment": (
        int(os.environ.get("RETROINFER_N_SEGMENT", "16"))
        if method == "retroinfer_author_common" else None
    ),
    "retroinfer_prefix_tokens": (
        int(os.environ.get("RETROINFER_PREFIX_TOKENS", "4"))
        if method == "retroinfer_reference_streaming" else None
    ),
    "retroinfer_recent_tokens": (
        int(os.environ.get("RETROINFER_RECENT_TOKENS", "64"))
        if method == "retroinfer_reference_streaming" else None
    ),
    "retroinfer_update_segment": (
        int(os.environ.get("RETROINFER_UPDATE_SEGMENT", "1024"))
        if method == "retroinfer_reference_streaming" else None
    ),
    "retroinfer_average_cluster_size": (
        int(os.environ.get("RETROINFER_AVERAGE_CLUSTER_SIZE", "16"))
        if method in {"retroinfer_reference_streaming",
                      "retroinfer_author_common"} else None
    ),
    "retroinfer_estimation_ratio": (
        float(os.environ.get("RETROINFER_ESTIMATION_RATIO", "0.232"))
        if method in {"retroinfer_reference_streaming",
                      "retroinfer_author_common"} else None
    ),
    "retroinfer_kmeans_iters": (
        int(os.environ.get("RETROINFER_KMEANS_ITERS", "10"))
        if method in {"retroinfer_reference_streaming",
                      "retroinfer_author_common"} else None
    ),
    "backing_store": (
        ("author_native_cpu_pinned"
         if method in {"pariskv_official", "magicpig_author_common",
                       "infllm_author_common"} else
         "exact_postrope_cpu_pinned"
         if method == "pqcache_author_common" else
         "exact_postrope_cpu_pinned"
         if os.environ.get("STREAMING_OFFLOAD", "0") == "1"
         else "exact_postrope_gpu")
        if "streaming" in method or method in {
            "pariskv_official", "pariskv_author_common",
            "pqcache_author_common", "magicpig_author_common",
            "infllm_author_common"
        } else None
    ),
    "streaming_gather_backend": (
        os.environ.get("STREAMING_GATHER_BACKEND", "auto")
        if os.environ.get("STREAMING_OFFLOAD", "0") == "1" else None
    ),
    "streaming_temporal_block_reuse": (
        os.environ.get("STREAMING_OFFLOAD", "0") == "1"
        and os.environ.get("STREAMING_GATHER_BACKEND", "auto") in {"auto", "uva"}
        and (
            method == "quest_streaming"
            or method.startswith("adaptive_centroid_lse_streaming")
        )
    ),
    "streaming_router_backend": (
        os.environ.get("STREAMING_ROUTER_BACKEND", "triton")
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "streaming_refine_factor": (
        float(os.environ.get("STREAMING_REFINE_FACTOR", "1"))
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "streaming_refine_candidate_ratio": (
        float(os.environ["STREAMING_REFINE_CANDIDATE_RATIO"])
        if method.startswith("adaptive_centroid_lse_streaming")
        and os.environ.get("STREAMING_REFINE_CANDIDATE_RATIO") else None
    ),
    "streaming_refine_tokens": (
        os.environ.get("STREAMING_REFINE_TOKENS", "0") == "1"
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "streaming_refine_sketch_rank": (
        int(os.environ.get("STREAMING_REFINE_SKETCH_RANK", "0"))
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "streaming_refine_sketch_bits": (
        int(os.environ.get("STREAMING_REFINE_SKETCH_BITS", "4"))
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "streaming_refine_sketch_basis": (
        os.environ.get("STREAMING_REFINE_SKETCH_BASIS", "query_aware")
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "streaming_max_components": (
        int(os.environ["STREAMING_MAX_COMPONENTS"])
        if method.startswith("adaptive_centroid_lse_streaming")
        and os.environ.get("STREAMING_MAX_COMPONENTS") else (
            int(chunk) if method.startswith("adaptive_centroid_lse_streaming") else None
        )
    ),
    "streaming_compact_metadata": (
        os.environ.get(
            "STREAMING_COMPACT_METADATA", "1" if querymean_default else "0"
        ) == "1"
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "streaming_center_bits": (
        int(os.environ.get(
            "STREAMING_CENTER_BITS", "8" if querymean_default else "16"
        ))
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "rmsnorm_backend": os.environ.get("SHADOWKV_RMSNORM_BACKEND", "torch"),
    "adaptive_active_mean_centers": (
        None if center_allocation in {
            "tail_relative_rate_distortion",
            "tail_absolute_rate_distortion",
            "tail_absolute_marginal",
            "tail_mass_rate_distortion",
            "tail_distortion_threshold", "tail_group_distortion_target",
            "tail_demand_adaptive_rate_distortion",
        } else 1.0 + float(os.environ.get(
            "ADAPTIVE_LSE_EXTRA_FRACTION", "0.25"
        ))
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "self_lse_cost": (
        os.environ.get("SHADOWKV_SELF_LSE_COST", default_self_lse_cost)
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "self_lse_cost_beta": (
        float(os.environ.get("SHADOWKV_SELF_LSE_COST_BETA", "4"))
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "center_dispersion_correction": (
        os.environ.get(
            "SHADOWKV_CENTER_DISPERSION_CORRECTION",
            "0" if querymean_default else "1",
        ) == "1"
        if method.startswith("adaptive_centroid_lse_streaming") else None
    ),
    "robust_trim_fraction": (
        float(os.environ.get("SHADOWKV_ROBUST_TRIM_FRACTION", "0.25"))
        if center_placement == "robust_trimmed"
        else None
    ),
    "tail_cvar_fraction": (
        float(os.environ.get("SHADOWKV_TAIL_CVAR_FRACTION", "0.25"))
        if center_allocation in {
            "tail_cvar", "anchor_tail_cvar", "density_tail_cvar",
            "tail_rate_distortion", "tail_relative_rate_distortion",
            "tail_absolute_rate_distortion", "tail_absolute_marginal",
            "tail_mass_rate_distortion", "tail_distortion_threshold",
            "tail_group_distortion_target",
            "tail_demand_adaptive_rate_distortion",
        }
        else None
    ),
    "tail_gap_correction_scale": (
        float(os.environ.get(
            "SHADOWKV_TAIL_GAP_CORRECTION_SCALE", default_gap_correction
        ))
        if center_allocation in {
            "tail_cvar", "anchor_tail_cvar", "density_tail_cvar",
            "tail_rate_distortion", "tail_relative_rate_distortion",
            "tail_absolute_rate_distortion", "tail_absolute_marginal",
            "tail_mass_rate_distortion", "tail_distortion_threshold",
            "tail_group_distortion_target",
            "tail_demand_adaptive_rate_distortion",
        }
        else None
    ),
    "relative_rate_distortion_penalty": (
        float(os.environ.get("SHADOWKV_RELATIVE_RD_PENALTY", "0.25"))
        if center_allocation == "tail_relative_rate_distortion" else None
    ),
    "absolute_rate_distortion_penalty": (
        float(os.environ.get(
            "SHADOWKV_ABSOLUTE_RD_PENALTY", default_absolute_rd_penalty
        ))
        if center_allocation in {
            "tail_absolute_rate_distortion", "tail_absolute_marginal",
            "tail_mass_rate_distortion"
        } else None
    ),
    "distortion_tolerance": (
        float(os.environ.get("SHADOWKV_DISTORTION_TOLERANCE", "1.0"))
        if center_allocation == "tail_distortion_threshold" else None
    ),
    "group_distortion_target": (
        float(os.environ.get("SHADOWKV_GROUP_DISTORTION_TARGET", "1.0"))
        if center_allocation == "tail_group_distortion_target" else None
    ),
    "demand_low_penalty": (
        float(os.environ.get("SHADOWKV_DEMAND_LOW_PENALTY", "1.5"))
        if center_allocation == "tail_demand_adaptive_rate_distortion" else None
    ),
    "demand_high_penalty": (
        float(os.environ.get("SHADOWKV_DEMAND_HIGH_PENALTY", "2.5"))
        if center_allocation == "tail_demand_adaptive_rate_distortion" else None
    ),
    "demand_threshold": (
        float(os.environ.get("SHADOWKV_DEMAND_THRESHOLD", "1.75"))
        if center_allocation == "tail_demand_adaptive_rate_distortion" else None
    ),
    "qanchor_weight": (
        float(os.environ.get("SHADOWKV_QANCHOR_WEIGHT", "0.25"))
        if center_allocation == "anchor_tail_cvar" else None
    ),
    "density_page_tokens": (
        int(os.environ.get("SHADOWKV_DENSITY_PAGE_TOKENS", "1024"))
        if os.environ.get("SHADOWKV_CENTER_ALLOCATION") == "density_tail_cvar"
        else None
    ),
    "density_reference_count": (
        int(os.environ.get("SHADOWKV_DENSITY_REFERENCE_COUNT", "64"))
        if os.environ.get("SHADOWKV_CENTER_ALLOCATION") == "density_tail_cvar"
        else None
    ),
    "density_kappa": (
        float(os.environ.get("SHADOWKV_DENSITY_KAPPA", "8"))
        if os.environ.get("SHADOWKV_CENTER_ALLOCATION") == "density_tail_cvar"
        else None
    ),
    "density_shrinkage": (
        float(os.environ.get("SHADOWKV_DENSITY_SHRINKAGE", "0.5"))
        if os.environ.get("SHADOWKV_CENTER_ALLOCATION") == "density_tail_cvar"
        else None
    ),
    "density_head_trust_power": (
        float(os.environ.get("SHADOWKV_DENSITY_HEAD_TRUST_POWER", "0"))
        if os.environ.get("SHADOWKV_CENTER_ALLOCATION") == "density_tail_cvar"
        else None
    ),
    "density_value_power": (
        float(os.environ.get("SHADOWKV_DENSITY_VALUE_POWER", "0"))
        if os.environ.get("SHADOWKV_CENTER_ALLOCATION") == "density_tail_cvar"
        else None
    ),
    "density_signal_mode": (
        os.environ.get("SHADOWKV_DENSITY_SIGNAL_MODE", "contrast")
        if os.environ.get("SHADOWKV_CENTER_ALLOCATION") == "density_tail_cvar"
        else None
    ),
    "density_weight_mode": (
        os.environ.get("SHADOWKV_DENSITY_WEIGHT_MODE", "symmetric")
        if os.environ.get("SHADOWKV_CENTER_ALLOCATION") == "density_tail_cvar"
        else None
    ),
    "exposure_reference_count": (
        int(os.environ.get("SHADOWKV_EXPOSURE_REFERENCE_COUNT", "64"))
        if os.environ.get("SHADOWKV_CENTER_ALLOCATION")
        == "exposure_angular_marginal"
        else None
    ),
    "head_allocation_tau": (
        float(os.environ["SHADOWKV_HEAD_ALLOC_TAU"])
        if os.environ.get("SHADOWKV_HEAD_ALLOC_TAU") else None
    ),
    "head_allocation_mode": os.environ.get("SHADOWKV_HEAD_ALLOC_MODE"),
    "head_allocation_coverage": float(
        os.environ.get("SHADOWKV_HEAD_ALLOC_COVERAGE", "0.9")
    ),
    "head_allocation_regularizer": float(
        os.environ.get("SHADOWKV_HEAD_ALLOC_REGULARIZER", "0")
    ),
    "head_allocation_entropy_prior_weight": float(
        os.environ.get("SHADOWKV_HEAD_ALLOC_ENTROPY_PRIOR_WEIGHT", "0")
    ),
    "head_allocation_residual_alpha": float(
        os.environ.get("SHADOWKV_HEAD_ALLOC_RESIDUAL_ALPHA", "1")
    ),
    "head_allocation_ema": float(
        os.environ.get("SHADOWKV_HEAD_ALLOC_EMA", "0")
    ),
    "machine": os.environ.get("SHADOWKV_MACHINE"), "gpu": gpu,
    "git_sha": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain")),
    "torch": torch.__version__, "transformers": transformers.__version__,
    "flash_attn": flash_attn.__version__,
}, open(out, "w"), indent=2)
PYEOF

echo "[cell] $CELL  gpu=$GPU  model=$MODEL_PATH"
SAMPLE_ARGS=()
SAMPLE_START=${SHADOWKV_SAMPLE_START:-}
SAMPLE_STOP=${SHADOWKV_SAMPLE_STOP:-}
if [[ -n "$SAMPLE_START" ]]; then
  SAMPLE_ARGS+=(--sample_start "$SAMPLE_START")
fi
if [[ -n "$SAMPLE_STOP" ]]; then
  SAMPLE_ARGS+=(--sample_stop "$SAMPLE_STOP")
fi
if [[ "$METHOD" == pariskv_official ]]; then
  [[ "$DATASET_NAME" == ruler/* ]] || {
    echo "pariskv_official runner currently accepts RULER only; use pariskv_author_common for $DATASET_NAME" >&2
    exit 2
  }
  OFFICIAL_ARGS=(
    "$PY" "$CODE/repro/shadowkv/eval_pariskv_official.py"
    --author-root "$PARISKV_AUTHOR_ROOT"
    --shadowkv-root "$SHADOWKV_DIR"
    --model "$MODEL_PATH"
    --datalen "$DATALEN"
    --task "$TASK"
    --final-topk "$BUDGET"
    --num-samples "$NUM_SAMPLES"
    --sink-size "${QUEST_PREFIX_TOKENS:-32}"
    --local-size "${STREAMING_RECENT_TOKENS:-32}"
    --out "$OUT_ROOT/$MODEL_KEY/$CELL.jsonl"
  )
  OFFICIAL_ARGS+=("${SAMPLE_ARGS[@]}")
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 "${OFFICIAL_ARGS[@]}"
  exit 0
fi
CMD=(
  "$PY" test/eval_acc.py
  --model_name "$MODEL_PATH"
  --datalen "$DATALEN"
  --method "$METHOD_ARG"
  --dataset_name "$DATASET_NAME"
  --num_samples "$NUM_SAMPLES"
  --out_root "$OUT_ROOT/$MODEL_KEY"
  --cell_name "$CELL"
)
CMD+=("${SAMPLE_ARGS[@]}")
CMD+=("${EXTRA[@]}")
if [[ -n "${SHADOWKV_MAX_NEW_TOKENS:-}" ]]; then
  CMD+=(--max_new_tokens "$SHADOWKV_MAX_NEW_TOKENS")
fi
if [[ -n "${SHADOWKV_GENERATION_TEMPERATURE:-}" ]]; then
  CMD+=(--generation_temperature "$SHADOWKV_GENERATION_TEMPERATURE")
fi
if [[ -n "${SHADOWKV_GENERATION_TOP_P:-}" ]]; then
  CMD+=(--generation_top_p "$SHADOWKV_GENERATION_TOP_P")
fi
if [[ -n "${SHADOWKV_GENERATION_TOP_K:-}" ]]; then
  CMD+=(--generation_top_k "$SHADOWKV_GENERATION_TOP_K")
fi
if [[ -n "${SHADOWKV_GENERATION_SEED:-}" ]]; then
  CMD+=(--generation_seed "$SHADOWKV_GENERATION_SEED")
fi
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 "${CMD[@]}"

if [ -n "${SHADOWKV_HEAD_ALLOC_TAU:-}" ]; then
  : "${STREAMING_HEAD_ALLOCATION_OUT:?adaptive run requires a traffic output}"
  [ -s "$STREAMING_HEAD_ALLOCATION_OUT" ] || {
    echo "ERROR: adaptive run produced no head-allocation audit" >&2
    exit 2
  }
  "$PY" - "$STREAMING_HEAD_ALLOCATION_OUT" \
    "$SHADOWKV_HEAD_ALLOC_TAU" "$SHADOWKV_HEAD_ALLOC_MODE" <<'PYEOF'
import json, math, sys
path, expected_tau, expected_mode = sys.argv[1:]
with open(path) as stream:
    audit = json.loads(stream.readlines()[-1])
if not math.isclose(float(audit["head_allocation_tau"]), float(expected_tau)):
    raise SystemExit("head-allocation audit has the wrong tau")
if audit["head_allocation_mode"] != expected_mode:
    raise SystemExit("head-allocation audit has the wrong mode")
PYEOF
fi
