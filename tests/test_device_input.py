#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for CUDA device-buffer frame input.

fp_image_view_t buffers (rgb/depth/mask) may be host OR device pointers; the
library resolves the location via unified virtual addressing. This test loads a
BOP YCB-V frame, stages it into GPU memory with raw libcudart (no torch/cupy
dependency), and checks that register and track produce bit-identical results
from device input and host input. The register path is deterministic, so any
difference is a real ingest bug.

Run inside the benchmark container (see tests/README.md):
  docker compose run --rm test tests/test_device_input.py
"""

from __future__ import annotations

import argparse
import ctypes as C
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import run_benchmark as rb  # noqa: E402

CUDA_MEMCPY_HOST_TO_DEVICE = 1


class Cuda:
    """Minimal libcudart shim: alloc, H2D copy, free."""

    def __init__(self) -> None:
        last_error = None
        for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
            try:
                self.rt = C.CDLL(name)
                break
            except OSError as exc:
                last_error = exc
        else:
            raise RuntimeError(f"could not load libcudart: {last_error}")
        self.rt.cudaMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
        self.rt.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
        self.rt.cudaFree.argtypes = [C.c_void_p]
        self.allocations: list[C.c_void_p] = []

    def check(self, status: int, what: str) -> None:
        if status != 0:
            raise RuntimeError(f"{what} failed with CUDA error {status}")

    def to_device(self, array: np.ndarray) -> C.c_void_p:
        ptr = C.c_void_p()
        self.check(self.rt.cudaMalloc(C.byref(ptr), array.nbytes), "cudaMalloc")
        self.allocations.append(ptr)
        self.check(
            self.rt.cudaMemcpy(ptr, array.ctypes.data_as(C.c_void_p), array.nbytes,
                               CUDA_MEMCPY_HOST_TO_DEVICE),
            "cudaMemcpy H2D")
        return ptr

    def free_all(self) -> None:
        for ptr in self.allocations:
            self.rt.cudaFree(ptr)
        self.allocations.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bop_path", default=os.environ.get("FP_BOP_PATH"),
                        required="FP_BOP_PATH" not in os.environ,
                        help="Path containing ycbv/.")
    parser.add_argument("--library", default=os.environ.get("FP_LIBRARY"),
                        help="Path to libfoundation_pose_nvidia.so (default: auto-discover).")
    parser.add_argument("--refine-model-path",
                        default=os.environ.get("FP_REFINE_MODEL_PATH"),
                        required="FP_REFINE_MODEL_PATH" not in os.environ,
                        help="Path to refiner_net.onnx.")
    parser.add_argument("--score-model-path",
                        default=os.environ.get("FP_SCORE_MODEL_PATH"),
                        required="FP_SCORE_MODEL_PATH" not in os.environ,
                        help="Path to score_net.onnx.")
    parser.add_argument("--engine-cache-dir",
                        default=os.environ.get(
                            "FP_ENGINE_CACHE_DIR",
                            str(Path(__file__).resolve().parents[1] / "engine_cache")),
                        help="Directory for TensorRT plan files.")
    parser.add_argument("--scene-id", type=int, default=48, help="YCB-V test scene.")
    parser.add_argument("--image-id", type=int, default=1, help="Register frame image ID.")
    parser.add_argument("--obj-id", type=int, default=1, help="Object ID in the frame.")
    parser.add_argument("--mesh-unit-scale", type=float, default=0.001,
                        help="YCB-V PLY meshes are in millimeters.")
    return parser.parse_args()


def make_view(width: int, height: int, rgb_ptr, depth_ptr, mask_ptr, k: np.ndarray) -> rb.ImageView:
    view = rb.ImageView()
    view.width, view.height = width, height
    view.rgb_u8 = C.cast(rgb_ptr, C.POINTER(C.c_uint8))
    view.depth_m = C.cast(depth_ptr, C.POINTER(C.c_float))
    view.mask_u8 = C.cast(mask_ptr, C.POINTER(C.c_uint8)) if mask_ptr else None
    view.k_row_major = k.ctypes.data_as(C.POINTER(C.c_float))
    return view


def pose_of(result: rb.PoseResult) -> np.ndarray:
    return np.array(result.pose_row_major, dtype=np.float32).reshape(4, 4)


def main() -> int:
    args = parse_args()
    lib = rb.load_library(Path(args.library) if args.library else None)
    layout = rb.validate_dataset(Path(args.bop_path), None, Path("/tmp"))
    if layout.errors:
        print(f"dataset validation failed: {layout.errors}")
        return 1
    cuda = Cuda()
    err = C.create_string_buffer(512)

    config = rb.Config()
    lib.fp_default_config(C.byref(config))
    config.n_hypotheses = int(os.environ.get("FP_N_HYPOTHESES", config.n_hypotheses))
    config.max_image_width, config.max_image_height = 1280, 720

    opt = rb.CreateOptions()
    opt.cad_path = str(layout.models_dir / f"obj_{args.obj_id:06d}.ply").encode()
    opt.device_id = 0
    opt.mesh_unit_scale = args.mesh_unit_scale
    opt.refine_model_path = args.refine_model_path.encode()
    opt.score_model_path = args.score_model_path.encode()
    opt.engine_cache_dir = args.engine_cache_dir.encode()

    handle = lib.fp_create(C.byref(opt), C.byref(config), err, len(err))
    assert handle, f"fp_create failed: {err.value.decode()}"
    handle = C.c_void_p(handle)
    lib.fp_prepare(handle, config.n_hypotheses, err, len(err))

    # Host frame (register frame with mask) + the following frame for track.
    rgb, depth, mask, k = rb.load_frame(layout, args.scene_id, args.image_id, args.obj_id, True)
    rgb = np.ascontiguousarray(rgb)
    depth = np.ascontiguousarray(depth.astype(np.float32))
    mask = np.ascontiguousarray(mask)
    k = np.ascontiguousarray(k.astype(np.float32))
    h, w = rgb.shape[:2]

    scene_camera = rb.read_json(layout.dataset_root / "test" / f"{args.scene_id:06d}"
                                / "scene_camera.json")
    next_im = next(i for i in sorted(int(x) for x in scene_camera) if i > args.image_id)
    rgb2, depth2, _, k2 = rb.load_frame(layout, args.scene_id, next_im, args.obj_id, False)
    rgb2 = np.ascontiguousarray(rgb2)
    depth2 = np.ascontiguousarray(depth2.astype(np.float32))
    k2 = np.ascontiguousarray(k2.astype(np.float32))

    # Device copies of both frames.
    d_rgb, d_depth, d_mask = cuda.to_device(rgb), cuda.to_device(depth), cuda.to_device(mask)
    d_rgb2, d_depth2 = cuda.to_device(rgb2), cuda.to_device(depth2)

    host_reg = make_view(w, h, rgb.ctypes.data_as(C.c_void_p),
                         depth.ctypes.data_as(C.c_void_p),
                         mask.ctypes.data_as(C.c_void_p), k)
    dev_reg = make_view(w, h, d_rgb, d_depth, d_mask, k)
    host_trk = make_view(w, h, rgb2.ctypes.data_as(C.c_void_p),
                         depth2.ctypes.data_as(C.c_void_p), None, k2)
    dev_trk = make_view(w, h, d_rgb2, d_depth2, None, k2)

    result = rb.PoseResult()

    # --- Register: host baseline, then device input, must be bit-identical. ---
    rc = lib.fp_register_frame(handle, C.byref(host_reg), -1, -1, C.byref(result), err, len(err))
    assert rc == 0, f"host register failed: {err.value.decode()}"
    host_reg_pose, host_reg_score = pose_of(result), result.score

    rc = lib.fp_track_frame(handle, C.byref(host_trk), -1, C.byref(result), err, len(err))
    assert rc == 0, f"host track failed: {err.value.decode()}"
    host_trk_pose = pose_of(result)

    rc = lib.fp_register_frame(handle, C.byref(dev_reg), -1, -1, C.byref(result), err, len(err))
    assert rc == 0, f"device register failed: {err.value.decode()}"
    dev_reg_pose, dev_reg_score = pose_of(result), result.score
    reg_delta = np.abs(dev_reg_pose - host_reg_pose).max()
    print(f"register: host score={host_reg_score:.4f} device score={dev_reg_score:.4f} "
          f"max|host-device| pose delta={reg_delta:.2e}")
    assert dev_reg_score == host_reg_score, "register score differs for device input"
    assert reg_delta == 0.0, f"register pose differs for device input ({reg_delta})"

    # --- Track from device buffers (state re-seeded by the device register). ---
    rc = lib.fp_track_frame(handle, C.byref(dev_trk), -1, C.byref(result), err, len(err))
    assert rc == 0, f"device track failed: {err.value.decode()}"
    trk_delta = np.abs(pose_of(result) - host_trk_pose).max()
    print(f"track:    max|host-device| pose delta={trk_delta:.2e}")
    assert trk_delta == 0.0, f"track pose differs for device input ({trk_delta})"

    lib.fp_destroy(handle)
    cuda.free_all()
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
