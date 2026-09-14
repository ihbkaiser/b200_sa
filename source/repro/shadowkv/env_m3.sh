#!/usr/bin/env bash
# Machine 3 (8 x L40): user-run ShadowKV environment.
# This launcher is portable after cloning shadowkv-research under $BASE.
#
#   BASE=/home/zhufangzhou/workspace/sheruifeng/baonn/baonn
#   cd "$BASE/shadowkv-research"
#   source repro/shadowkv/env_m3.sh

export SHADOWKV_MACHINE=m3
export SHADOWKV_BASE=${SHADOWKV_BASE:-/home/zhufangzhou/workspace/sheruifeng/baonn/baonn}

# zsh does not set BASH_SOURCE, and sourcing this from zsh used to resolve one
# directory too high and abort.  $0 is the sourced path under zsh, and the
# working directory is the last resort.
if [ -n "${BASH_SOURCE:-}" ]; then
  _SHADOWKV_M3_ENV=$(readlink -f "${BASH_SOURCE[0]}")
else
  _SHADOWKV_M3_ENV=$(readlink -f "$0")
fi
export CODE=$(cd "$(dirname "$_SHADOWKV_M3_ENV")/../.." && pwd -P)
unset _SHADOWKV_M3_ENV
if [ ! -d "$CODE/ShadowKV" ] && [ -d "$PWD/ShadowKV" ]; then
  export CODE=$PWD
fi
if [ ! -d "$CODE/ShadowKV" ]; then
  echo "env_m3.sh resolved an invalid checkout: $CODE" >&2
  echo "  chạy từ trong thư mục repo, hoặc đặt CODE tường minh" >&2
  return 1 2>/dev/null || exit 1
fi
export SHADOWKV_DIR=$CODE/ShadowKV
export PY=${PY:-$SHADOWKV_BASE/envs/shadowkv/bin/python}

export SHADOWKV_LLAMA32_PATH=${SHADOWKV_LLAMA32_PATH:-$SHADOWKV_BASE/models/Llama-3.2-3B-Instruct}
export SHADOWKV_QWEN3_PATH=${SHADOWKV_QWEN3_PATH:-$SHADOWKV_BASE/models/Qwen3-4B-Instruct-2507}

export SHADOWKV_LONGBENCH_PATH=${SHADOWKV_LONGBENCH_PATH:-$SHADOWKV_BASE/datasets/LongBench}
export SHADOWKV_LONGBENCH_V2_PATH=${SHADOWKV_LONGBENCH_V2_PATH:-$SHADOWKV_BASE/datasets/LongBench-v2}
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
    echo "[env_m3] KE THUA  $_name=$_have" >&2
    echo "[env_m3]          mac dinh repo la $_want" >&2
    echo "[env_m3]          unset $_name roi source lai neu khong co y" >&2
  fi
  eval "export $_name=\${$_name:-\$_want}"
}
_shadowkv_pin SHADOWKV_AIME25_PATH  "$CODE/repro/shadowkv/data/reasoning/kvpress_aime25_local"
_shadowkv_pin SHADOWKV_MATH500_PATH "$CODE/repro/shadowkv/data/reasoning/kvpress_math500_local"
_shadowkv_pin SHADOWKV_GPQA_PATH    "$CODE/repro/shadowkv/data/reasoning/kvpress_gpqa"
export PARISKV_AUTHOR_ROOT=${PARISKV_AUTHOR_ROOT:-$SHADOWKV_BASE/ParisKV-official}
export RETROINFER_AUTHOR_ROOT=${RETROINFER_AUTHOR_ROOT:-$SHADOWKV_BASE/RetrievalAttention}
export SHADOWKV_THIRD_PARTY_ROOT=${SHADOWKV_THIRD_PARTY_ROOT:-$SHADOWKV_BASE/third_party}

export HF_HOME=${HF_HOME:-$SHADOWKV_BASE/hf_cache}
export NLTK_DATA=${NLTK_DATA:-$SHADOWKV_BASE/nltk_data}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
_shadowkv_pin SHADOWKV_RESULTS_ROOT "$SHADOWKV_BASE/results_shadowkv"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export SHADOWKV_POOL_GPUS=${SHADOWKV_POOL_GPUS:-"0,1,2,3,4,5,6,7"}
export SHADOWKV_GPU_POLICY="GPU0-7 eligible L40 48GB; user must confirm live ownership before launch"
if [[ -x /usr/local/cuda-12.0/bin/nvcc ]]; then
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.0}
else
  export CUDA_HOME=${CUDA_HOME:-$SHADOWKV_BASE/envs/shadowkv}
fi

# Sourcing this twice in one shell does NOT reset it: every path below is
# ${VAR:-default}, so a value already exported wins over the default -- which
# is the point when you override on purpose, and a trap when the value is left
# over from an earlier run. On 2026-09-14 a stale SHADOWKV_RESULTS_ROOT from a
# smoke made a campaign report read an empty directory and look like data loss.
# So say out loud what was resolved; `unset` the variable and source again to
# get the default back.
echo "[env_m3] code    $CODE" >&2
echo "[env_m3] results $SHADOWKV_RESULTS_ROOT" >&2
echo "[env_m3] data    $(dirname "$SHADOWKV_AIME25_PATH")" >&2
