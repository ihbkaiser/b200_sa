#!/usr/bin/env bash
# Machine 2 (sepc810): standalone ShadowKV/Quest research environment.
#
#   cd /home/nbnguyen/sparse_attention
#   source repro/shadowkv/env_m2.sh

export SHADOWKV_MACHINE=m2

export CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export SHADOWKV_DIR=$CODE/ShadowKV
export PARISKV_AUTHOR_ROOT=${PARISKV_AUTHOR_ROOT:-/home/nbnguyen/ParisKV-official}
export PQCACHE_AUTHOR_ROOT=${PQCACHE_AUTHOR_ROOT:-/home/nbnguyen/upstream-kv-methods/PQCache}
export MAGICPIG_AUTHOR_ROOT=${MAGICPIG_AUTHOR_ROOT:-/home/nbnguyen/upstream-kv-methods/MagicPIG}
export RETROINFER_AUTHOR_ROOT=${RETROINFER_AUTHOR_ROOT:-/home/nbnguyen/upstream-kv-methods/RetrievalAttention}
export PY=/home/nbnguyen/envs/shadowkv/bin/python
export PATH=$(dirname "$PY"):$PATH

export SHADOWKV_LLAMA32_PATH=/home/nbnguyen/models/Llama-3.2-3B-Instruct
export SHADOWKV_QWEN3_PATH=/home/nbnguyen/models/Qwen3-4B-Instruct-2507

export SHADOWKV_LONGBENCH_PATH=/home/nbnguyen/datasets/LongBench
export SHADOWKV_LONGBENCH_V2_PATH=/home/nbnguyen/datasets/LongBench-v2
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
    echo "[env_m2] KE THUA  $_name=$_have" >&2
    echo "[env_m2]          mac dinh repo la $_want" >&2
    echo "[env_m2]          unset $_name roi source lai neu khong co y" >&2
  fi
  eval "export $_name=\${$_name:-\$_want}"
}
_shadowkv_pin SHADOWKV_AIME25_PATH  "$CODE/repro/shadowkv/data/reasoning/kvpress_aime25_local"
_shadowkv_pin SHADOWKV_MATH500_PATH "$CODE/repro/shadowkv/data/reasoning/kvpress_math500_local"
_shadowkv_pin SHADOWKV_GPQA_PATH    "$CODE/repro/shadowkv/data/reasoning/kvpress_gpqa"

# Machine 2's large filesystem is /home (7.3 TiB total), not /storage.
export HF_HOME=/home/nbnguyen/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
_shadowkv_pin SHADOWKV_RESULTS_ROOT "/home/nbnguyen/sparse_attention_results"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# GPU0 is an NVIDIA T400 4GB and is never eligible. GPU1--4 are 24GB Ada.
export SHADOWKV_POOL_GPUS=${SHADOWKV_POOL_GPUS:-"1,2,3,4"}
export SHADOWKV_GPU_POLICY="GPU0 forbidden (NVIDIA T400 4GB); GPU1-4 eligible RTX 4500 Ada 24GB"

export CUDA_HOME=/home/nbnguyen/envs/shadowkv

# Sourcing this twice in one shell does NOT reset it: every path below is
# ${VAR:-default}, so a value already exported wins over the default -- which
# is the point when you override on purpose, and a trap when the value is left
# over from an earlier run. On 2026-09-14 a stale SHADOWKV_RESULTS_ROOT from a
# smoke made a campaign report read an empty directory and look like data loss.
# So say out loud what was resolved; `unset` the variable and source again to
# get the default back.
echo "[env_m2] code    $CODE" >&2
echo "[env_m2] results $SHADOWKV_RESULTS_ROOT" >&2
echo "[env_m2] data    $(dirname "$SHADOWKV_AIME25_PATH")" >&2
