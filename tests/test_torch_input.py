#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for GPU tensor input through the Python wrapper.

RgbdFrame accepts any array exposing `__cuda_array_interface__` (torch CUDA
tensors, CuPy, Numba) for rgb/depth/mask. This test builds a deterministic
synthetic frame, runs register/track with host numpy arrays, then with torch
CUDA tensors of the same content, and asserts bit-identical results. Also
covers mixed host/GPU buffers and the dtype/contiguity validation errors.

Requires torch (present in the project container's system python, not the
/opt/bench venv):
  docker compose run --rm --entrypoint bash test -c \
    'PYTHONPATH=python/src python3 tests/test_torch_input.py'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))
from foundation_pose_nvidia import (  # noqa: E402
    Estimator, EstimatorOptions, RgbdFrame, RuntimeConfig)

WIDTH, HEIGHT = 640, 480
CAD_PATH = os.environ.get("FP_CAD_PATH",
                          "data/BOP_datasets/ycbv/models/obj_000001.ply")


def synthetic_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.full((HEIGHT, WIDTH, 3), 96, dtype=np.uint8)
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    k = np.array([[WIDTH, 0, WIDTH / 2], [0, WIDTH, HEIGHT / 2], [0, 0, 1]],
                 dtype=np.float32)
    box = min(WIDTH, HEIGHT) // 3
    x0, y0 = (WIDTH - box) // 2, (HEIGHT - box) // 2
    rgb[y0:y0 + box, x0:x0 + box] = (180, 130, 90)
    depth[y0:y0 + box, x0:x0 + box] = 0.7
    mask[y0:y0 + box, x0:x0 + box] = 255
    return rgb, depth, mask, k


def main() -> int:
    assert torch.cuda.is_available(), "torch CUDA is required for this test"

    options = EstimatorOptions.from_env(CAD_PATH, mesh_unit_scale=0.001)
    n_hypotheses = int(os.environ.get("FP_N_HYPOTHESES", "252"))
    config = RuntimeConfig(max_image_width=WIDTH, max_image_height=HEIGHT,
                           n_hypotheses=n_hypotheses)

    rgb, depth, mask, k = synthetic_frame()
    t_rgb = torch.from_numpy(rgb).cuda().contiguous()
    t_depth = torch.from_numpy(depth).cuda().contiguous()
    t_mask = torch.from_numpy(mask).cuda().contiguous()

    with Estimator(options, config, prepare_batch=n_hypotheses) as est:
        # Host baseline.
        host_reg = est.register(RgbdFrame(rgb, depth, k, mask))
        host_trk = est.track(RgbdFrame(rgb, depth, k))

        # Full-GPU frame (torch CUDA tensors), must match the host path exactly.
        gpu_reg = est.register(RgbdFrame(t_rgb, t_depth, k, t_mask))
        gpu_trk = est.track(RgbdFrame(t_rgb, t_depth, k))
        reg_delta = np.abs(gpu_reg.pose - host_reg.pose).max()
        trk_delta = np.abs(gpu_trk.pose - host_trk.pose).max()
        print(f"torch-cuda register: score host={host_reg.score:.4f} "
              f"gpu={gpu_reg.score:.4f} max pose delta={reg_delta:.2e}")
        print(f"torch-cuda track:    max pose delta={trk_delta:.2e}")
        assert gpu_reg.score == host_reg.score and reg_delta == 0.0
        assert trk_delta == 0.0

        # Mixed host/GPU buffers (GPU rgb+depth, host mask).
        mix_reg = est.register(RgbdFrame(t_rgb, t_depth, k, mask))
        assert np.abs(mix_reg.pose - host_reg.pose).max() == 0.0
        print("mixed host/GPU buffers: ok")

        # Validation: wrong dtype and non-contiguous GPU tensors must raise.
        try:
            RgbdFrame(t_rgb, t_depth.double(), k)
            raise AssertionError("float64 GPU depth was not rejected")
        except ValueError as exc:
            print(f"dtype validation: ok ({exc})")
        try:
            RgbdFrame(t_rgb.permute(1, 0, 2), t_depth, k)
            raise AssertionError("non-contiguous GPU rgb was not rejected")
        except ValueError as exc:
            print(f"contiguity validation: ok ({exc})")

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
