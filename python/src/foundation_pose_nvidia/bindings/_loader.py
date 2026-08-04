# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover and load libfoundation_pose_nvidia.so."""

from __future__ import annotations

import os
from pathlib import Path

from foundation_pose_nvidia.exceptions import LibraryNotFoundError


def repo_root() -> Path:
    """Best-effort repo root for default library discovery."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "CMakeLists.txt").is_file() and (parent / "include").is_dir():
            return parent
    return here.parents[4]


def candidate_libraries(root: Path | None = None) -> list[Path]:
    root = root or repo_root()
    names = [
        "libfoundation_pose_nvidia.so",
        "libfoundation_pose_nvidia.dylib",
        "foundation_pose_nvidia.dll",
    ]
    search_dirs = [
        root / "build",
        root / "build" / "Release",
        root / "build" / "Debug",
        root / "out" / "build",
    ]
    return [directory / name for directory in search_dirs for name in names]


def resolve_library_path(path: Path | str | None = None) -> Path:
    if path is not None:
        selected = Path(path).expanduser().resolve()
    elif env_path := os.environ.get("FP_LIBRARY"):
        selected = Path(env_path).expanduser().resolve()
    else:
        selected = next((p for p in candidate_libraries() if p.exists()), None)

    if selected is None or not selected.is_file():
        searched = "\n  ".join(str(p) for p in candidate_libraries())
        raise LibraryNotFoundError(f"Could not find shared library. Searched:\n  {searched}")
    return selected
