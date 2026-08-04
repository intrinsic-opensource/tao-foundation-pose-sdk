#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the YCB-V benchmark harness INSIDE a container (built from
# benchmark.Dockerfile as image `fp-bench`, or the base pytorch image as a
# fallback). Ensures benchmark/eval deps + bop_toolkit, then forwards all args
# to benchmarks/run_benchmark.py. Usage (inside docker run):
#   bash docker_benchmark.sh --benchmark --mode register --bop_path ... ...
set -euo pipefail
cd "$(dirname "$0")/.."
export HOME=/tmp

# Prefer the baked benchmark venv (fp-bench image, has GL/EGL libs + eval deps);
# otherwise create a fallback venv in the mount (works for inference, but BOP
# VSD eval needs the GL libs that only the fp-bench image installs).
if [[ -x /opt/bench/bin/python ]]; then
  VENV=/opt/bench
else
  VENV=/work/.venv_bench
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "=== creating fallback venv (no system GL libs; eval may fail) ==="
    python -m venv --system-site-packages "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q numpy==1.26.4 opencv-python-headless==4.11.0.86 \
      scipy==1.17.1 imageio==2.37.3 pypng==0.20220715.0 pillow==12.2.0 \
      vispy==0.16.2 PyOpenGL==3.1.10 pytz==2026.2 scikit-image==0.26.0
  fi
fi

# bop_toolkit (editable) for --evaluate. Idempotent.
"$VENV/bin/python" -c "import bop_toolkit_lib" 2>/dev/null \
  || "$VENV/bin/pip" install -q -e /work/data/BOP_datasets/bop_toolkit --no-deps

# Do NOT forward ${LD_LIBRARY_PATH:-}: the container runtime environment
# contains paths that conflict with the benchmark's own library layout.
# libcuda.so.1 is found correctly via ldconfig — the NVIDIA container runtime
# registers the forward-compat lib (lib.real-nvgpu/libcuda.so.1.1, version
# 595.56) in the container's ldconfig at startup, bridging the CUDA 13.2
# userspace API that libcudart requires down to the host's nvgpu kernel (580.00).
# It does not need to be in LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="$PWD/build:/usr/lib/$(uname -m)-linux-gnu"

# Fill in dataset/model/library/cache paths from FP_* env vars (set by
# compose.yaml) unless the corresponding flag was passed explicitly.
ARGS=("$@")
add_default() {
  local flag="$1" value="$2"
  [[ -z "$value" ]] && return 0
  for arg in "${ARGS[@]}"; do
    [[ "$arg" = "$flag" ]] && return 0
  done
  ARGS+=("$flag" "$value")
}
add_default --bop_path          "${FP_BOP_PATH:-}"
add_default --refine-model-path "${FP_REFINE_MODEL_PATH:-}"
add_default --score-model-path  "${FP_SCORE_MODEL_PATH:-}"
add_default --library           "${FP_LIBRARY:-}"
add_default --engine-cache-dir  "${FP_ENGINE_CACHE_DIR:-}"

exec "$VENV/bin/python" benchmarks/run_benchmark.py "${ARGS[@]}"
