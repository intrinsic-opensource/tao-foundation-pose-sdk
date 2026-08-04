#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build (on demand, via CMake) and run sample_app_cpp. This is the entrypoint
# for the `sample_app_cpp` compose service, and also works from an interactive
# shell inside the project container.
#
# It compiles only if the binary is missing (or REBUILD=1). The example links
# the prebuilt libfoundation_pose_nvidia.so, so build the library first:
#   docker compose run --rm build
#
# Usage:
#   bash example/cpp/run.sh                  # BOP YCB-V mode
#   bash example/cpp/run.sh --use_synthesize # synthetic mode
#
# Any args are forwarded to the binary.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
build_dir="$here/build"
bin="$build_dir/sample_app_cpp"

lib="$repo/build/libfoundation_pose_nvidia.so"
if [[ ! -f "$lib" ]]; then
  echo "error: $lib not found — build the library first: docker compose run --rm build" >&2
  exit 1
fi

if [[ ! -x "$bin" || "${REBUILD:-0}" == "1" ]]; then
  echo "=== configuring + building sample_app_cpp ==="
  cmake -S "$here" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$build_dir" -j"${BUILD_JOBS:-4}"
fi

echo "=== running sample_app_cpp ==="
exec "$bin" "$@"
