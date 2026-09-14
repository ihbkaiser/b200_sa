#!/usr/bin/env bash
# m1 (fatchoy) environment for the ShadowKV / Quest campaign.
# Source this before anything else; every path below has been seen in an `ls`.
#
#   source repro/shadowkv/env_m1.sh

export SHADOWKV_MACHINE=m1

# --- code -------------------------------------------------------------------
export CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export SHADOWKV_DIR=$CODE/ShadowKV
export PARISKV_AUTHOR_ROOT=${PARISKV_AUTHOR_ROOT:-/home/baonn/ParisKV-official}
export PQCACHE_AUTHOR_ROOT=${PQCACHE_AUTHOR_ROOT:-/home/baonn/upstream-kv-methods/PQCache}
export MAGICPIG_AUTHOR_ROOT=${MAGICPIG_AUTHOR_ROOT:-/home/baonn/upstream-kv-methods/MagicPIG}
export RETROINFER_AUTHOR_ROOT=${RETROINFER_AUTHOR_ROOT:-/home/baonn/upstream-kv-methods/RetrievalAttention}
export PY=$HOME/miniconda3/envs/shadowkv/bin/python
export PATH=$(dirname "$PY"):$PATH

# --- models (local, no hub access needed) -----------------------------------
export SHADOWKV_LLAMA32_PATH=/storage/baonn/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
export SHADOWKV_QWEN3_PATH=/storage/baonn/qwen3-4b-instruct-2507

# --- benchmark datasets (the same local copies used by KVPress) ------------
export SHADOWKV_LONGBENCH_PATH=/storage/baonn/huggingface/hub/datasets--Xnhyacinth--LongBench/snapshots/2e9ade51ebf45d98942056c0716234f9d5d257d5
export SHADOWKV_LONGBENCH_V2_PATH=/storage/baonn/huggingface/hub/datasets--simonjegou--LongBench-v2/snapshots/5f14e86aebef1ba64a519eb6864e52b1985b9c94
# The three reasoning benchmarks are vendored in the repo and shared by every
# machine -- see repro/shadowkv/data/reasoning/README.md. They used to be
# per-machine copies, and the copies drifted: m1 read the AIME parquet that
# carries the \boxed{} instruction while m2/m3/m4 read the one that does not,
# which made their AIME cells silently unmergeable. Override only on purpose.

# --- pin a path, and say so when the shell already had a different one -------
# Every path here is ${VAR:-default}, so a value already exported wins. That is
# right when the override is deliberate and silent poison when it is left over
# from an earlier run: on 2026-09-14 a stale SHADOWKV_MATH500_PATH survived a
# `git pull` that moved the data into the repo, and the campaign refused to
# start with a path nobody had typed. eval, not ${!name}, because this file is
# sourced from zsh as often as from bash and zsh spells indirection ${(P)name}.
_shadowkv_pin() {
  local _name=$1 _want=$2 _have
  eval "_have=\${$_name:-}"
  if [ -n "$_have" ] && [ "$_have" != "$_want" ]; then
    echo "[env_m1] KE THUA  $_name=$_have" >&2
    echo "[env_m1]          mac dinh repo la $_want" >&2
    echo "[env_m1]          unset $_name roi source lai neu khong co y" >&2
  fi
  eval "export $_name=\${$_name:-\$_want}"
}
_shadowkv_pin SHADOWKV_AIME25_PATH  "$CODE/repro/shadowkv/data/reasoning/kvpress_aime25_local"
_shadowkv_pin SHADOWKV_MATH500_PATH "$CODE/repro/shadowkv/data/reasoning/kvpress_math500_local"
_shadowkv_pin SHADOWKV_GPQA_PATH    "$CODE/repro/shadowkv/data/reasoning/kvpress_gpqa"

# --- M51 offline anchor banks (repro/shadowkv/m51_fit_anchors.py) ----------
# Built from the certified-sparse calibration query bank, which exists for
# Qwen3 only; Llama has no bank yet, so m51 cells for llama32 refuse to run
# rather than silently falling back to something else.
export SHADOWKV_M51_ANCHORS_QWEN3=${SHADOWKV_M51_ANCHORS_QWEN3:-/storage/baonn/sparse_attention_results/_m51/anchors_qwen3_m8.npz}
export SHADOWKV_M51_ANCHORS_LLAMA32=${SHADOWKV_M51_ANCHORS_LLAMA32:-}

# --- caches / results (m1's big mount is /storage, 1.8T) --------------------
export HF_HOME=/storage/baonn/huggingface
_shadowkv_pin SHADOWKV_RESULTS_ROOT "/storage/baonn/sparse_attention_results"

# --- GPU --------------------------------------------------------------------
# nvidia-smi always numbers by PCI bus; CUDA defaults to FASTEST_FIRST. The pool
# uses one number for both, so pin the ordering or a roster picked to avoid a
# card will quietly include it.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# GPU0 is the user's card and sits in the badly-cooled corner: short tests only,
# never a long pool.
export SHADOWKV_POOL_GPUS=${SHADOWKV_POOL_GPUS:-"1,2,3,4,5,6,7"}
export SHADOWKV_GPU_POLICY="GPU0 short tests only; GPU1-7 eligible unless live process/maintenance says otherwise"

# --- build toolchain (only needed to rebuild the CUDA kernels) --------------
export CUDA_HOME=$HOME/miniconda3/envs/shadowkv

# Sourcing this twice in one shell does NOT reset it: every path below is
# ${VAR:-default}, so a value already exported wins over the default -- which
# is the point when you override on purpose, and a trap when the value is left
# over from an earlier run. On 2026-09-14 a stale SHADOWKV_RESULTS_ROOT from a
# smoke made a campaign report read an empty directory and look like data loss.
# So say out loud what was resolved; `unset` the variable and source again to
# get the default back.
echo "[env_m1] code    $CODE" >&2
echo "[env_m1] results $SHADOWKV_RESULTS_ROOT" >&2
echo "[env_m1] data    $(dirname "$SHADOWKV_AIME25_PATH")" >&2
