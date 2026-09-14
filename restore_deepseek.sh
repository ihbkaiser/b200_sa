#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL="$ROOT/model/DeepSeek-R1-Distill-Llama-8B"
PY=${PY:-python3}
KEEP_COMPRESSED=${KEEP_COMPRESSED:-1}

command -v zstd >/dev/null || { echo "zstd is required" >&2; exit 2; }
[ -d "$MODEL" ] || { echo "DeepSeek model directory not found: $MODEL" >&2; exit 2; }

for archive in "$MODEL"/model-*.safetensors.zst; do
  [ -f "$archive" ] || { echo "missing compressed DeepSeek shard" >&2; exit 2; }
  raw=${archive%.zst}
  tmp="$raw.partial"
  echo "[restore] $archive -> $raw"
  zstd -T0 -d -q -f "$archive" -o "$tmp"
  mv -f "$tmp" "$raw"
  if [ "$KEEP_COMPRESSED" = 0 ]; then
    "$PY" - "$archive" <<'PY'
import sys
from pathlib import Path
Path(sys.argv[1]).unlink()
PY
  fi
done

echo "[done] DeepSeek raw safetensors restored"
