# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class FoundationPoseError(RuntimeError):
    """Base error for the Python wrapper."""


class LibraryNotFoundError(FoundationPoseError):
    """Raised when libfoundation_pose_nvidia.so cannot be located."""
