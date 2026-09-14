#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-python3}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

command -v "$PY" >/dev/null || { echo "Python not found: $PY" >&2; exit 2; }
[ -d "$ROOT/source/ShadowKV/3rdparty/cutlass/include" ] || {
  echo "CUTLASS is missing from the artifact" >&2; exit 2;
}

if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  if [ "${OFFLINE_INSTALL:-0}" = 1 ]; then
    echo "[deps] installing bundled PyTorch 2.11.0+cu128"
    "$PY" -m pip install --no-index --find-links "$ROOT/wheels" \
      'torch==2.11.0+cu128'
  else
    echo "PyTorch is missing. Install a B200-compatible torch wheel first, or set OFFLINE_INSTALL=1" >&2
    exit 2
  fi
fi

echo "[preflight] Python: $($PY --version 2>&1)"
"$PY" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; this artifact requires a B200 GPU")
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"[preflight] GPU: {name}, compute capability {cap[0]}.{cap[1]}")
print(f"[preflight] torch: {torch.__version__}, CUDA runtime: {torch.version.cuda}")
if cap < (10, 0):
    raise SystemExit("This setup script is for B200/Blackwell (compute capability >= 10.0)")
PY

if [ "${SKIP_DEPS:-0}" != 1 ]; then
  if [ "${OFFLINE_INSTALL:-0}" = 1 ]; then
    [ -d "$ROOT/wheels" ] || { echo "wheelhouse missing: $ROOT/wheels" >&2; exit 2; }
    "$PY" -m pip install --no-index --find-links "$ROOT/wheels" -r "$ROOT/requirements-b200.txt"
  else
    "$PY" -m pip install -r "$ROOT/requirements-b200.txt"
  fi
fi

if ! "$PY" -c 'from flash_attn import flash_attn_with_kvcache' >/dev/null 2>&1; then
  if [ "${SKIP_FLASH_ATTN:-0}" = 1 ]; then
    echo "[warning] flash-attn is not installed (SKIP_FLASH_ATTN=1)" >&2
  elif [ "${OFFLINE_INSTALL:-0}" = 1 ]; then
    echo "[build] installing flash-attn from the local wheelhouse/source archive"
    "$PY" -m pip install --no-index --no-build-isolation \
      --find-links "$ROOT/wheels" flash-attn
  else
    "$PY" -m pip install --no-build-isolation flash-attn
  fi
fi

echo "[build] compiling ShadowKV CUDA extension for Blackwell sm_100"
cd "$ROOT/source/ShadowKV"
export CUDA_HOME
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-10.0}
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
"$PY" setup.py build_ext --inplace

"$PY" - <<'PY'
import torch
from kernels import shadowkv
from flash_attn import flash_attn_with_kvcache
print("[verify] ShadowKV CUDA extension: OK")
print("[verify] flash-attn: OK")
print("[verify] device:", torch.cuda.get_device_name(0))
PY
echo "[done] B200 build is ready"
