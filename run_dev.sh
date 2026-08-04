#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Platform-detecting Docker Compose wrapper.
# Automatically selects the right overlay based on host architecture:
#   aarch64  →  docker compose -f compose.yaml -f compose.thor.yaml
#   x86_64   →  docker compose  (compose.override.yaml auto-loaded for dGPU)
#
# Usage (identical on both platforms):
#   ./run_dev.sh run --rm build
#   ./run_dev.sh run --rm cli --smoke --mode register ...
#   ./run_dev.sh run --rm bench --benchmark --mode register --max-targets 75
#   ./run_dev.sh run --rm test
#   ./run_dev.sh up -d dev && ./run_dev.sh exec dev bash
#   ./run_dev.sh config     # inspect the fully-merged compose config

set -euo pipefail

# Pre-create host-side mount directories as the current user so Docker daemon
# (running as root) does not create them with root ownership on first run.
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
mkdir -p "${FP_DATA_DIR:-./data}" \
         "${FP_WEIGHTS_DIR:-./weights}" \
         "${FP_ENGINE_CACHE_DIR:-./engine_cache}"

ARCH=$(uname -m)

if [[ "$ARCH" = "aarch64" ]]; then
    exec docker compose -f compose.yaml -f compose.thor.yaml "$@"
else
    exec docker compose "$@"   # compose.override.yaml auto-loaded for dGPU
fi
