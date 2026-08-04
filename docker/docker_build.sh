#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build nvidia-foundationpose-runtime INSIDE the pytorch:26.05 container.
# CUDA 13 / TensorRT 10.13 / nvdiffrast CUDA rasterizer, arch sm_120 (Blackwell).
set -euo pipefail
cd "$(dirname "$0")/.."
export HOME=/tmp

# --- Pinned nvdiffrast commit (validated with this runtime: build + smoke test +
#     BOP YCB-V benchmark). Override with NVDIFFRAST_COMMIT=<sha> if needed.
# The optional nvdiffrast renderer plugin is only fetched/built when
# WITH_NVDIFFRAST=1 (default off: the built-in CUDA rasterizer needs nothing).
WITH_NVDIFFRAST="${WITH_NVDIFFRAST:-0}"
NVDIFFRAST_COMMIT="${NVDIFFRAST_COMMIT:-253ac4fcea7de5f396371124af597e6cc957bfae}"
if [[ "${WITH_NVDIFFRAST}" = "1" ]] && [[ ! -e external/nvdiffrast/csrc/common/rasterize.cu ]]; then
  echo "=== fetching nvdiffrast @ ${NVDIFFRAST_COMMIT} ==="
  mkdir -p external/nvdiffrast
  git -C external/nvdiffrast init -q
  git -C external/nvdiffrast remote add origin https://github.com/NVlabs/nvdiffrast.git 2>/dev/null || true
  git -C external/nvdiffrast fetch -q --depth 1 origin "${NVDIFFRAST_COMMIT}"
  git -C external/nvdiffrast checkout -q FETCH_HEAD
fi

# --- Patch: nvdiffrast framework.h only defines NVDR_CHECK / NVDR_CHECK_CUDA_ERROR
#     under -DNVDR_TORCH (PyTorch builds). This runtime compiles nvdiffrast's CUDA
#     rasterizer standalone (-DNVDR_STANDALONE, no torch), so those macros are
#     missing -> compile error in cudaraster/impl/*. Provide a non-torch fallback.
#     Idempotent: only appended once (safe across fresh nvdiffrast clones).
FW="external/nvdiffrast/csrc/common/framework.h"
if [[ "${WITH_NVDIFFRAST}" = "1" ]] && ! grep -q "NVDR_STANDALONE_FP_RUNTIME_MACROS" "$FW"; then
  echo "=== patching $FW for standalone (non-torch) macros ==="
  cat >> "$FW" <<'PATCH'

//------------------------------------------------------------------------
// NVDR_STANDALONE_FP_RUNTIME_MACROS
// Standalone (non-PyTorch) fallback for foundation_pose_nvidia runtime, which
// compiles nvdiffrast's CUDA rasterizer as a plain CUDA/C++ library. Upstream
// framework.h defines these only under NVDR_TORCH. Keyed on !NVDR_TORCH so it
// never conflicts with the PyTorch/TensorFlow builds.
#ifndef NVDR_TORCH
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define NVDR_CHECK(COND, ERR) do { if (!(COND)) { std::fprintf(stderr, "nvdiffrast check failed: %s\n", ERR); std::abort(); } } while(0)
#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) do { cudaError_t err_ = (CUDA_CALL); if (err_ != cudaSuccess) { std::fprintf(stderr, "Cuda error: %s [%s]\n", cudaGetErrorString(err_), #CUDA_CALL); std::abort(); } } while(0)
#endif
//------------------------------------------------------------------------
PATCH
fi

# --- Patch 2: nvdiffrast's cudaraster assumes the PyTorch build pulls in
#     <algorithm> (min/max), <cstring> (memset/memcpy), and a glog-style
#     LOG()/INFO. The standalone build has none. Defs.hpp is included by every
#     cudaraster translation unit, so inject the prelude there. Idempotent.
DEFS="external/nvdiffrast/csrc/common/cudaraster/impl/Defs.hpp"
if [[ "${WITH_NVDIFFRAST}" = "1" ]] && ! grep -q "NVDR_STANDALONE_FP_RUNTIME_PRELUDE" "$DEFS"; then
  echo "=== patching $DEFS for standalone prelude (algorithm/cstring/LOG) ==="
  cat >> "$DEFS" <<'PATCH'

//------------------------------------------------------------------------
// NVDR_STANDALONE_FP_RUNTIME_PRELUDE  (keyed on !NVDR_TORCH; inert otherwise)
#ifndef NVDR_TORCH
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#ifndef INFO
#define INFO 0
#endif
#ifndef LOG
namespace CR { struct FpNullLog { template <typename T> FpNullLog& operator<<(const T&) { return *this; } }; }
#define LOG(level) ::CR::FpNullLog()
#endif
#endif
//------------------------------------------------------------------------
PATCH
fi

echo "=== toolchain ==="
nvcc --version | tail -2
cmake --version | head -1
echo "=== configure ==="
# Resolve CMAKE_CUDA_ARCHITECTURES, in order of precedence:
#   1. Explicit CUDA_ARCHITECTURES env, already in CMake format
#      (e.g. "120", "native" on Thor, or a list like "75;80;86").
#   2. The container-provided CUDA_ARCH_LIST / TORCH_CUDA_ARCH_LIST
#      (NGC PyTorch images export e.g. "7.5 8.0 8.6 9.0 10.0 12.0[+PTX]"),
#      converted to CMake format: 7.5 -> 75, 10.0 -> 100, 12.0 -> 120.
#   3. Fallback: 120 (Blackwell).
if [[ -z "${CUDA_ARCHITECTURES:-}" ]]; then
  ARCH_SRC="${CUDA_ARCH_LIST:-${TORCH_CUDA_ARCH_LIST:-}}"
  if [[ -n "${ARCH_SRC}" ]]; then
    CUDA_ARCHITECTURES="$(printf '%s\n' ${ARCH_SRC} | sed -e 's/+PTX$//' -e 's/\.//g' \
                          | grep -E '^[0-9]+a?$' | paste -sd';' -)"
  fi
  CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-120}"
fi
echo "CUDA architectures: ${CUDA_ARCHITECTURES}"
NVDR_FLAGS=(-DFOUNDATION_POSE_WITH_NVDIFFRAST=OFF)
if [[ "${WITH_NVDIFFRAST}" = "1" ]]; then
  NVDR_FLAGS=(-DFOUNDATION_POSE_WITH_NVDIFFRAST=ON -DNVDIFFRAST_ROOT="$PWD/external/nvdiffrast")
fi
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  "${NVDR_FLAGS[@]}" \
  -DTENSORRT_ROOT=/usr
echo "=== build (-j${BUILD_JOBS:-4}) ==="
cmake --build build -j"${BUILD_JOBS:-4}"
echo "=== artifacts ==="
ls -la build/*.so build/foundation_pose_nvidia_cli 2>/dev/null || ls -la build/
