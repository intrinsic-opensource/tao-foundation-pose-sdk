#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download the FoundationPose deployable ONNX weights from NGC
# (nvidia/tao/foundationpose:deployable_v1.0) and place them under the weights
# directory with the names the rest of this repo expects:
#
#   <WEIGHTS_DIR>/refiner_net.onnx   (RefineNet)
#   <WEIGHTS_DIR>/score_net.onnx     (ScoreNet)
#
# Usage:
#   scripts/download_weights.sh [weights_dir]
#
# The model is published publicly on NGC, so this downloads anonymously by
# default — no API key required.
#
# Environment:
#   WEIGHTS_DIR   Target directory (default: ./weights, or $FP_WEIGHTS_DIR if set).
#   NGC_API_KEY   Optional. Not needed for this public model. If set, it is sent
#                 directly as a Bearer token (the current `nvapi-*` key format).
#   FORCE=1       Re-download even if both target files already exist.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_ORG="nvidia"
MODEL_TEAM="tao"
MODEL_NAME="foundationpose"
MODEL_VERSION="deployable_v1.0"
ZIP_URL="https://api.ngc.nvidia.com/v2/models/${MODEL_ORG}/${MODEL_TEAM}/${MODEL_NAME}/versions/${MODEL_VERSION}/zip"

WEIGHTS_DIR="${1:-${WEIGHTS_DIR:-${FP_WEIGHTS_DIR:-./weights}}}"
REFINER_DST="${WEIGHTS_DIR}/refiner_net.onnx"
SCORE_DST="${WEIGHTS_DIR}/score_net.onnx"

# --- Skip if already present (unless FORCE=1) --------------------------------
if [[ "${FORCE:-0}" != "1" && -f "$REFINER_DST" && -f "$SCORE_DST" ]]; then
  echo "weights already present:"
  echo "  $REFINER_DST"
  echo "  $SCORE_DST"
  echo "(set FORCE=1 to re-download)"
  exit 0
fi

for tool in wget unzip; do
  command -v "$tool" >/dev/null 2>&1 || { echo "error: '$tool' is required but not installed" >&2; exit 1; }
done

mkdir -p "$WEIGHTS_DIR"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
ZIP_PATH="${WORK_DIR}/foundationpose_${MODEL_VERSION}.zip"

# --- Auth: the model is public, so none is needed by default. --------------
# If NGC_API_KEY is set, send it directly as a Bearer token. Current NGC keys
# (`nvapi-*`) are used as-is; the old "ApiKey -> token exchange" flow via
# authn.nvidia.com has been discontinued and is no longer used.
AUTH_HEADER=()
if [[ -n "${NGC_API_KEY:-}" ]]; then
  echo "=== using NGC_API_KEY as a Bearer token ==="
  AUTH_HEADER=(--header="Authorization: Bearer ${NGC_API_KEY}")
fi

# --- Download ----------------------------------------------------------------
echo "=== downloading ${MODEL_ORG}/${MODEL_TEAM}/${MODEL_NAME}:${MODEL_VERSION} ==="
echo "    $ZIP_URL"
if ! wget "${AUTH_HEADER[@]}" -O "$ZIP_PATH" "$ZIP_URL"; then
  echo "error: download failed. This model is public and needs no key, so this is" >&2
  echo "       most likely a network/proxy issue. Verify connectivity to" >&2
  echo "       api.ngc.nvidia.com and retry." >&2
  exit 1
fi

# --- Extract -----------------------------------------------------------------
echo "=== extracting ==="
EXTRACT_DIR="${WORK_DIR}/extracted"
mkdir -p "$EXTRACT_DIR"
unzip -o -q "$ZIP_PATH" -d "$EXTRACT_DIR"

# --- Locate the two ONNX files by name pattern (names vary across packages) ---
find_onnx() {
  # $1 = space-separated case-insensitive keywords; return first matching .onnx
  local pattern="$1" f
  while IFS= read -r f; do
    local base; base="$(basename "$f" | tr '[:upper:]' '[:lower:]')"
    for kw in $pattern; do
      if [[ "$base" == *"$kw"* ]]; then echo "$f"; return 0; fi
    done
  done < <(find "$EXTRACT_DIR" -type f -iname '*.onnx' | sort)
  return 1
}

REFINER_SRC="$(find_onnx "refin refine refiner")" || true
SCORE_SRC="$(find_onnx "score scorer")" || true

if [[ -z "${REFINER_SRC:-}" || -z "${SCORE_SRC:-}" ]]; then
  echo "error: could not identify both ONNX files in the package. Found:" >&2
  find "$EXTRACT_DIR" -type f -iname '*.onnx' -printf '  %p\n' >&2 || true
  echo "Rename them to refiner_net.onnx / score_net.onnx under $WEIGHTS_DIR manually." >&2
  exit 1
fi

# --- Install with the canonical names ----------------------------------------
cp -f "$REFINER_SRC" "$REFINER_DST"
cp -f "$SCORE_SRC" "$SCORE_DST"

echo "=== done ==="
echo "  RefineNet: $(basename "$REFINER_SRC")  ->  $REFINER_DST"
echo "  ScoreNet : $(basename "$SCORE_SRC")  ->  $SCORE_DST"
ls -la "$REFINER_DST" "$SCORE_DST"
