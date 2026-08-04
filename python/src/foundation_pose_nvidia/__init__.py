# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Python bindings for the FoundationPose NVIDIA C ABI.

Public API
----------
- :mod:`foundation_pose_nvidia.core` — high-level estimator API
- :mod:`foundation_pose_nvidia.bindings` — low-level ctypes bindings
"""

from foundation_pose_nvidia._version import __version__
from foundation_pose_nvidia.bindings import FPLibrary, load_library
from foundation_pose_nvidia.core import (
    Estimator,
    EstimatorOptions,
    MultiObjectEstimator,
    MultiObjectPoseEstimate,
    ObjectSpec,
    PoseEstimate,
    Precision,
    RegisterOptions,
    RgbdFrame,
    RuntimeConfig,
)
from foundation_pose_nvidia.exceptions import FoundationPoseError, LibraryNotFoundError

__all__ = [
    "Estimator",
    "EstimatorOptions",
    "FPLibrary",
    "FoundationPoseError",
    "LibraryNotFoundError",
    "MultiObjectEstimator",
    "MultiObjectPoseEstimate",
    "ObjectSpec",
    "PoseEstimate",
    "Precision",
    "RegisterOptions",
    "RgbdFrame",
    "RuntimeConfig",
    "__version__",
    "load_library",
]
