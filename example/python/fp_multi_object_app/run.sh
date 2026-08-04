#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
lib="$repo/build/libfoundation_pose_nvidia.so"

if [[ ! -f "$lib" ]]; then
  echo "error: $lib not found; build the library first: docker compose run --rm build" >&2
  exit 1
fi

export PYTHONPATH="$repo/python/src:$here:$repo${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$repo/build:${LD_LIBRARY_PATH:-}"

python_bin=python3
if [[ -x /opt/bench/bin/python ]]; then
  python_bin=/opt/bench/bin/python
  "$python_bin" -m pip install -q -e "$repo/python"
fi

exec "$python_bin" "$here/main.py" "$@"
