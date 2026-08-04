#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Usage: run_sample.sh [app] [app-flags...]
#
#   (no args)                 run cpp then python single-object (both)
#   cpp [flags...]            run cpp only, forwarding [flags...] to example/cpp/run.sh
#   python [flags...]         run Python single-object example only
#   python_multi [flags...]   run Python multi-object example only
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

APP="${1:-both}"
shift || true             # shift off APP (safe even if $# was 0 via the default)

case "$APP" in
  cpp)
    bash example/cpp/run.sh "$@"
    ;;
  python)
    bash example/python/fp_single_object_app/run.sh "$@"
    ;;
  python_multi|python-multi)
    bash example/python/fp_multi_object_app/run.sh "$@"
    ;;
  both|"")
    bash example/cpp/run.sh "$@"
    bash example/python/fp_single_object_app/run.sh "$@"
    ;;
  *)
    echo "usage: run_sample.sh [cpp|python|python_multi|both] [flags...]" >&2
    exit 1
    ;;
esac
