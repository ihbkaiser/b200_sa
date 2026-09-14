#!/usr/bin/env bash
# Reproducibly prepare the authors' ParisKV runtime as an external dependency.
# Their source is intentionally not vendored: the inspected snapshot has no
# license file.  This script pins the tested commit and applies only local-model
# compatibility fixes required by Transformers 4.55.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PY=${PY:-$(command -v python)}
AUTHOR_ROOT=${1:-${PARISKV_AUTHOR_ROOT:-}}
if [[ -z "$AUTHOR_ROOT" ]]; then
  echo "usage: PARISKV_AUTHOR_ROOT=/path/to/ParisKV PY=/path/to/python $0" >&2
  exit 2
fi
AUTHOR_COMMIT=db7ad7f59ebe5670f9f2e1092b1b689802dda7c5
FHT_COMMIT=1cc807efbd6cc001df359822d60bf6052dd66859
THIRD_PARTY_ROOT=${SHADOWKV_THIRD_PARTY_ROOT:-$(dirname "$AUTHOR_ROOT")/third_party}
FHT_ROOT=$THIRD_PARTY_ROOT/fast-hadamard-transform

"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA-enabled PyTorch is required"
assert torch.version.cuda == "12.4", f"tested CUDA is 12.4, found {torch.version.cuda}"
print(f"OK torch={torch.__version__} cuda={torch.version.cuda}")
PY

if [[ ! -d "$AUTHOR_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$AUTHOR_ROOT")"
  git clone https://github.com/amy-77/ParisKV.git "$AUTHOR_ROOT"
fi
test "$(git -C "$AUTHOR_ROOT" remote get-url origin)" = \
  "https://github.com/amy-77/ParisKV.git" || {
    echo "unexpected ParisKV origin under $AUTHOR_ROOT" >&2; exit 1;
  }
git -C "$AUTHOR_ROOT" fetch origin "$AUTHOR_COMMIT"
git -C "$AUTHOR_ROOT" checkout --detach "$AUTHOR_COMMIT"

COMPAT=$SCRIPT_DIR/pariskv_official_compat.patch
if git -C "$AUTHOR_ROOT" apply --check "$COMPAT" 2>/dev/null; then
  git -C "$AUTHOR_ROOT" apply "$COMPAT"
elif ! git -C "$AUTHOR_ROOT" apply --reverse --check "$COMPAT" 2>/dev/null; then
  echo "ParisKV compatibility patch neither applies nor is already present" >&2
  exit 1
fi

"$PY" -m pip install "flashinfer-python==0.2.4"
if ! "$PY" -c \
  'import importlib.metadata, fast_hadamard_transform; assert importlib.metadata.version("fast-hadamard-transform") == "1.1.0"' \
  2>/dev/null; then
  mkdir -p "$THIRD_PARTY_ROOT"
  if [[ ! -d "$FHT_ROOT/.git" ]]; then
    git clone --recursive https://github.com/Dao-AILab/fast-hadamard-transform.git \
      "$FHT_ROOT"
  fi
  git -C "$FHT_ROOT" fetch origin "$FHT_COMMIT"
  git -C "$FHT_ROOT" checkout --detach "$FHT_COMMIT"
  git -C "$FHT_ROOT" submodule update --init --recursive
  CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$PY")")} \
    "$PY" -m pip install --no-build-isolation "$FHT_ROOT"
fi

export PARISKV_AUTHOR_ROOT=$AUTHOR_ROOT
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$PY")")}
export PATH=$CUDA_HOME/bin:$PATH
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST=$("$PY" -c \
    'import torch; p=torch.cuda.get_device_properties(0); print(f"{p.major}.{p.minor}")')
fi
export TORCH_CUDA_ARCH_LIST
"$PY" - <<'PY'
import fast_hadamard_transform, flash_attn, flashinfer, torch
x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
y = fast_hadamard_transform.hadamard_transform(x)
assert y.shape == x.shape and torch.isfinite(y).all()
print("PARISKV OFFICIAL RUNTIME READY")
print(f"flash_attn={flash_attn.__version__} flashinfer={flashinfer.__version__}")
PY
