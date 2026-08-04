#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download the BOP YCB-V dataset into the layout this repo expects:
#
#   <DATA_DIR>/BOP_datasets/
#     ycbv/          (models/, test/, test_targets_bop19.json, camera*.json, ...)
#     bop_toolkit/   (github.com/thodan/bop_toolkit, for --evaluate)
#
# The dataset ships as separate zips so you only fetch what you need:
#   base     ~small  camera params, test_targets_bop19.json, dataset info   (always useful)
#   models   ~small  the CAD meshes (models/, models_eval/)   <- enough for the CLI smoke test
#   test     ~0.7GB  the bop19 test split: 75 sparse keyframes/scene         <- register/BOP19 benchmark
#   test_all ~15GB   the FULL test split: every consecutive video frame      <- video (dense) tracking benchmark
#   toolkit  git     bop_toolkit checkout                      <- needed for --evaluate
#
# `test` gives the sparse BOP19 keyframes (gaps of tens of frames — fine for
# register / one-shot AR, poor for tracking). `test_all` gives every consecutive
# frame of each test sequence (gap = 1), which is what real frame-to-frame
# tracking needs; use it with `run_benchmark.py --mode tracking --track-all-frames`.
#
# Usage:
#   scripts/download_bop_ycbv.sh [parts...]
#     parts: any of  base models test test_all toolkit  (default: base models toolkit)
#            'all'  = base models test toolkit          (sparse test split)
#            'video'= base models test_all toolkit      (full-frame test split for tracking)
#            'smoke'= models              (just the meshes, for the CLI smoke test)
#
# Examples:
#   scripts/download_bop_ycbv.sh smoke                 # just the CAD meshes
#   scripts/download_bop_ycbv.sh                       # base + models + toolkit (no big test split)
#   scripts/download_bop_ycbv.sh all                   # everything, incl. the large test split
#
# Environment:
#   DATA_DIR   Target root (default: ./data, or $FP_DATA_DIR if set). BOP_datasets/
#              is created underneath it.
#   FORCE=1    Re-download/re-extract even if the target already looks present.
set -euo pipefail
cd "$(dirname "$0")/.."

# BOP moved its dataset hosting from bop.felk.cvut.cz to the HuggingFace mirror
# (huggingface.co/datasets/bop-benchmark/ycbv). Override BASE_URL to point
# elsewhere (e.g. the old server) if needed.
BASE_URL="${BASE_URL:-https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main}"
TOOLKIT_URL="https://github.com/thodan/bop_toolkit.git"

DATA_DIR="${DATA_DIR:-${FP_DATA_DIR:-./data}}"
BOP_DIR="${DATA_DIR}/BOP_datasets"
YCBV_DIR="${BOP_DIR}/ycbv"

# --- Resolve which parts to fetch --------------------------------------------
parts=("$@")
if [[ ${#parts[@]} -eq 0 ]]; then
  parts=(base models toolkit)
elif [[ "${parts[*]}" == "all" ]]; then
  parts=(base models test toolkit)
elif [[ "${parts[*]}" == "video" ]]; then
  parts=(base models test_all toolkit)
elif [[ "${parts[*]}" == "smoke" ]]; then
  parts=(models)
fi

for tool in wget unzip git; do
  command -v "$tool" >/dev/null 2>&1 || { echo "error: '$tool' is required but not installed" >&2; exit 1; }
done

mkdir -p "$BOP_DIR"

# fetch_zip <name> <url> <sentinel-path-under-ycbv>
# Downloads and unzips so the contents always land under YCBV_DIR. The BOP zips
# are inconsistent: ycbv_base.zip has a top-level ycbv/ folder, while
# ycbv_models.zip / ycbv_test_bop19.zip unpack models/, test/, ... at the root
# (meant to be extracted *into* ycbv/). Detect which and place accordingly.
fetch_zip() {
  local name="$1" url="$2" sentinel="$3"
  if [[ "${FORCE:-0}" != "1" && -e "${YCBV_DIR}/${sentinel}" ]]; then
    echo "=== ${name}: already present (${YCBV_DIR}/${sentinel}) — skipping (FORCE=1 to redo) ==="
    return 0
  fi
  local tmp; tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" RETURN
  echo "=== downloading ${name} ==="
  echo "    ${url}"
  wget --content-disposition -O "${tmp}/${name}.zip" "${url}"
  echo "=== extracting ${name} ==="
  local ex="${tmp}/extracted"; mkdir -p "$ex"
  unzip -o -q "${tmp}/${name}.zip" -d "$ex"
  mkdir -p "$YCBV_DIR"
  if [[ -d "${ex}/ycbv" ]]; then
    # Zip carried the ycbv/ prefix — merge its contents into YCBV_DIR.
    cp -a "${ex}/ycbv/." "${YCBV_DIR}/"
  else
    # Zip's contents (models/, test/, ...) belong directly under ycbv/.
    cp -a "${ex}/." "${YCBV_DIR}/"
  fi
}

for part in "${parts[@]}"; do
  case "$part" in
    base)
      fetch_zip "ycbv_base" "${BASE_URL}/ycbv_base.zip" "test_targets_bop19.json"
      ;;
    models)
      fetch_zip "ycbv_models" "${BASE_URL}/ycbv_models.zip" "models/obj_000001.ply"
      ;;
    test)
      echo "note: the ycbv test split is several GB — this will take a while."
      fetch_zip "ycbv_test_bop19" "${BASE_URL}/ycbv_test_bop19.zip" "test"
      ;;
    test_all)
      echo "note: the FULL ycbv test split (every video frame) is ~15 GB — this will take a while."
      # Sentinel is a non-keyframe frame that exists only in the full split
      # (the bop19 subset ships sparse keyframes, so frame 000030 is absent there).
      fetch_zip "ycbv_test_all" "${BASE_URL}/ycbv_test_all.zip" "test/000048/rgb/000030.png"
      ;;
    toolkit)
      if [[ "${FORCE:-0}" != "1" && -d "${BOP_DIR}/bop_toolkit/.git" ]]; then
        echo "=== toolkit: already cloned (${BOP_DIR}/bop_toolkit) — skipping ==="
      else
        echo "=== cloning bop_toolkit ==="
        rm -rf "${BOP_DIR}/bop_toolkit"
        git clone --depth 1 "${TOOLKIT_URL}" "${BOP_DIR}/bop_toolkit"
      fi
      ;;
    *)
      echo "error: unknown part '$part' (want: base models test test_all toolkit | all | video | smoke)" >&2
      exit 2
      ;;
  esac
done

echo
echo "=== done. layout under ${BOP_DIR}: ==="
ls -la "$YCBV_DIR" 2>/dev/null || true
[[ -d "${BOP_DIR}/bop_toolkit" ]] && echo "  bop_toolkit/  (cloned)"
echo
echo "Point FP_DATA_DIR at '${DATA_DIR}' in your .env (currently the compose mount -> /work/data)."
