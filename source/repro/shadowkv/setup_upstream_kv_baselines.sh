#!/usr/bin/env bash
# Fetch the exact author repositories used by the PQCache/MagicPIG/InfLLM
# adapters.  Code travels through git; models, data and environments remain
# machine-local as required by the research-cluster workflow.
set -euo pipefail

: "${PY:?source repro/shadowkv/env_m1.sh (or the machine-specific env) first}"

ROOT=${SHADOWKV_UPSTREAM_ROOT:-${SHADOWKV_BASE:-$HOME}/upstream-kv-methods}
METHODS=",${UPSTREAM_KV_METHODS:-pqcache,magicpig,infllm},"
mkdir -p "$ROOT"

enabled() { [[ "$METHODS" == *",$1,"* ]]; }

fetch() {
  local name=$1 url=$2 commit=$3
  local dst="$ROOT/$name"
  if [ ! -d "$dst/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "$url" "$dst"
  fi
  git -C "$dst" fetch --depth 1 origin "$commit"
  if [ -n "$(git -C "$dst" status --porcelain)" ]; then
    echo "ERROR: author checkout is dirty: $dst" >&2
    exit 2
  fi
  git -C "$dst" checkout --detach "$commit"
}

enabled pqcache && fetch PQCache https://github.com/HugoZHL/PQCache.git \
  0b74e125207dc3f24da3bbaaf84e8a5f1d3b1828
enabled magicpig && fetch MagicPIG https://github.com/Infini-AI-Lab/MagicPIG.git \
  ac9aa36c866330ca6ad2ce342a7848d7df6f49bb
enabled infllm && fetch InfLLM https://github.com/thunlp/InfLLM.git \
  12b70798f56e56ebb23c53c7018091a3f540a028

# InfLLM is imported from source and has no extra compiled dependency beyond
# packages already used by ShadowKV.  MagicPIG's LSH and sparse-attention
# implementations are the author C++ extensions.
if enabled pqcache; then
  "$PY" -m pip install "kmeans-gpu==0.0.5" "scikit-learn==1.5.1" \
    "loguru>=0.7,<1" "pybind11>=2.6,<3"
  LFU_SRC=$ROOT/PQCache/vq_method/retrieval_based/lfu
  LFU_BUILD=$LFU_SRC/build
  PYBIND11_CMAKE=$($PY -m pybind11 --cmakedir)
  cmake -S "$LFU_SRC" -B "$LFU_BUILD" \
    -Dpybind11_DIR="$PYBIND11_CMAKE" \
    -DPYTHON_EXECUTABLE="$PY"
  cmake --build "$LFU_BUILD" -j "${PQCACHE_BUILD_JOBS:-4}"
fi
if enabled magicpig; then
  # Both released setup.py files unconditionally compile with -mavx512f.
  # Importing those wheels on an AVX2-only host terminates the interpreter
  # with SIGILL, so reject the host before building a misleading wheel.
  grep -qm1 -w avx512f /proc/cpuinfo || {
    echo "ERROR: official MagicPIG CPU kernels require AVX-512F" >&2
    exit 4
  }
  "$PY" -m pip install py-cpuinfo Cython
  "$PY" -m pip install -e "$ROOT/MagicPIG/library/lsh" --no-build-isolation
  "$PY" -m pip install -e "$ROOT/MagicPIG/library/sparse_attention" --no-build-isolation
fi

if enabled pqcache; then
  PQCACHE_AUTHOR_ROOT="$ROOT/PQCache" "$PY" - <<'PY'
import os, sys, torch, kmeans_gpu, types
sys.path.insert(0, os.environ["PQCACHE_AUTHOR_ROOT"])
name = "vq_method.retrieval_based.sparq_official.methods.ann_attention"
stub = types.ModuleType(name)
stub.MistralAttentionWithANN = object
stub.Settings = object
sys.modules[name] = stub
from vq_method.retrieval_based import cache_manager, pq_search
print("PQCACHE_NATIVE_COMPONENTS_READY", torch.__version__, cache_manager.lfucache.__file__, pq_search.__file__)
PY
fi
if enabled magicpig; then
  "$PY" -c 'import torch; from lsh import LSH; from sparse_attention_cpu import SparseAttentionServer; print("MAGICPIG_NATIVE_COMPONENTS_READY", torch.__version__)'
fi

cat <<EOF
UPSTREAM_BASELINES_READY=$ROOT methods=${METHODS#,}
export PQCACHE_AUTHOR_ROOT=$ROOT/PQCache
export MAGICPIG_AUTHOR_ROOT=$ROOT/MagicPIG
export INFLLM_AUTHOR_ROOT=$ROOT/InfLLM
EOF
