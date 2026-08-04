# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseEstimate:
    """Object pose returned by register/track."""

    pose: np.ndarray  # 4x4 row-major object-to-camera SE(3), meters
    score: float
    elapsed_s: float

    @classmethod
    def from_ctypes(cls, pose_row_major, score: float, elapsed_s: float) -> PoseEstimate:
        pose = np.asarray(pose_row_major, dtype=np.float32).reshape(4, 4).copy()
        return cls(pose=pose, score=float(score), elapsed_s=float(elapsed_s))
