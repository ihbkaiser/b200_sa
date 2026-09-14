#!/usr/bin/env bash
# Pack MATH500, AIME25 and GPQA into one bundle for a machine with no HF hub.
#
#   source repro/shadowkv/env_m1.sh
#   repro/shadowkv/make_reasoning_data_bundle.sh /storage/baonn/reasoning_data.tgz
#
# The three benchmarks are 360 kB of parquet in total, but WHICH parquet is not
# a detail: /storage/baonn/kvpress_aime25_local and .../kvpress_aime25_percontext
# are both 30 AIME problems and they are NOT the same benchmark -- percontext
# drops the "Remember to put your final answer within \boxed{}" instruction, so
# the extraction-based scorer reads a different number off the same model. This
# packs the variants env_m1.sh names, which are the ones every recorded
# reasoning number on this project was measured under.
#
# Verify on the far side with the printed sha256. Do NOT push this through the
# m2 reverse tunnel: that carries commands and status, never artifacts.
set -euo pipefail

: "${SHADOWKV_AIME25_PATH:?source repro/shadowkv/env_m1.sh first}"
: "${SHADOWKV_MATH500_PATH:?source repro/shadowkv/env_m1.sh first}"
: "${SHADOWKV_GPQA_PATH:?source repro/shadowkv/env_m1.sh first}"

OUT=${1:?usage: make_reasoning_data_bundle.sh <out.tgz>}
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

# The bundle's directory names are what env_mX.sh points at on the far side, so
# unpacking it under $BASE/datasets is the whole install.
cp -rL "$SHADOWKV_AIME25_PATH"  "$STAGE/kvpress_aime25_local"
cp -rL "$SHADOWKV_MATH500_PATH" "$STAGE/kvpress_math500_local"
cp -rL "$SHADOWKV_GPQA_PATH"    "$STAGE/kvpress_gpqa"

# Reproducible bytes: same input, same sha256, so the far side can be checked
# against this line months later.
tar --sort=name --owner=0 --group=0 --numeric-owner \
    --mtime='@0' -czf "$OUT" -C "$STAGE" \
    kvpress_aime25_local kvpress_math500_local kvpress_gpqa

echo "$OUT"
du -h "$OUT"
sha256sum "$OUT"
