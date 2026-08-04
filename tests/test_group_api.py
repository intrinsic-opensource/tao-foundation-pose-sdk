#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the fp_group_* multi-object C API.

Loads one BOP YCB-V frame that contains several objects, registers all of
them in a single fp_group_register_frame call, and checks that every pose is
identical to the one produced by the single-object fp_register_frame path on
the same estimator (the register path is deterministic, so poses must match
exactly). Then tracks all objects over the next few frames, comparing group
(concurrent) against serial single-handle wall time, and exercises the
borrowed-handle re-register path.

Run inside the benchmark container (see tests/README.md).
"""

from __future__ import annotations

import argparse
import ctypes as C
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import run_benchmark as rb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Path flags fall back to the FP_* env vars set by compose.yaml.
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
    parser.add_argument("--obj-ids", default="1,6,14,19,20",
                        help="Comma-separated object IDs present in the register frame.")
    parser.add_argument("--track-frames", type=int, default=6,
                        help="Number of subsequent frames to track.")
    parser.add_argument("--mesh-unit-scale", type=float, default=0.001,
                        help="YCB-V PLY meshes are in millimeters.")
    return parser.parse_args()


def load_group_symbols(lib: C.CDLL, object_count: int) -> None:
    mask_array = C.POINTER(C.c_uint8) * object_count
    lib.fp_group_create.restype = C.c_void_p
    lib.fp_group_create.argtypes = [C.POINTER(rb.CreateOptions), C.c_size_t,
                                    C.POINTER(rb.Config), C.c_char_p, C.c_size_t]
    lib.fp_group_size.restype = C.c_size_t
    lib.fp_group_size.argtypes = [C.c_void_p]
    lib.fp_group_handle.restype = C.c_void_p
    lib.fp_group_handle.argtypes = [C.c_void_p, C.c_size_t]
    lib.fp_group_prepare.restype = C.c_int
    lib.fp_group_prepare.argtypes = [C.c_void_p, C.c_int, C.c_char_p, C.c_size_t]
    lib.fp_group_register_frame.restype = C.c_int
    lib.fp_group_register_frame.argtypes = [C.c_void_p, C.POINTER(rb.ImageView), mask_array,
                                            C.c_int, C.c_int, C.POINTER(rb.PoseResult),
                                            C.POINTER(C.c_int), C.c_char_p, C.c_size_t]
    lib.fp_group_track_frame.restype = C.c_int
    lib.fp_group_track_frame.argtypes = [C.c_void_p, C.POINTER(rb.ImageView),
                                         C.POINTER(C.c_int), C.c_int,
                                         C.POINTER(rb.PoseResult), C.POINTER(C.c_int),
                                         C.c_char_p, C.c_size_t]
    lib.fp_group_destroy.restype = None
    lib.fp_group_destroy.argtypes = [C.c_void_p]


def make_view(rgb: np.ndarray, depth: np.ndarray, k: np.ndarray) -> rb.ImageView:
    view = rb.ImageView()
    view.width, view.height = rgb.shape[1], rgb.shape[0]
    view.rgb_u8 = rgb.ctypes.data_as(C.POINTER(C.c_uint8))
    view.depth_m = depth.ctypes.data_as(C.POINTER(C.c_float))
    view.mask_u8 = None
    view.k_row_major = k.ctypes.data_as(C.POINTER(C.c_float))
    return view


def contiguous_frame(layout: rb.DatasetLayout, scene_id: int, im_id: int, obj_id: int,
                     need_mask: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    rgb, depth, mask, k = rb.load_frame(layout, scene_id, im_id, obj_id, need_mask=need_mask)
    rgb = np.ascontiguousarray(rgb)
    depth = np.ascontiguousarray(depth.astype(np.float32))
    k = np.ascontiguousarray(k.astype(np.float32))
    if mask is not None:
        mask = np.ascontiguousarray(mask)
    return rgb, depth, mask, k


def pose_matrix(result: rb.PoseResult) -> np.ndarray:
    return np.array(result.pose_row_major, dtype=np.float32).reshape(4, 4)


def main() -> int:
    args = parse_args()
    obj_ids = [int(x) for x in args.obj_ids.split(",")]
    lib = rb.load_library(Path(args.library) if args.library else None)
    load_group_symbols(lib, len(obj_ids))

    layout = rb.validate_dataset(Path(args.bop_path), None, Path("/tmp"))
    if layout.errors:
        print(f"dataset validation failed: {layout.errors}")
        return 1
    err = C.create_string_buffer(512)

    config = rb.Config()
    lib.fp_default_config(C.byref(config))
    config.n_hypotheses = int(os.environ.get("FP_N_HYPOTHESES", config.n_hypotheses))
    config.max_image_width, config.max_image_height = 1280, 720

    opts = (rb.CreateOptions * len(obj_ids))()
    for i, obj_id in enumerate(obj_ids):
        opts[i].cad_path = str(layout.models_dir / f"obj_{obj_id:06d}.ply").encode()
        opts[i].device_id = 0
        opts[i].mesh_unit_scale = args.mesh_unit_scale
        opts[i].refine_model_path = args.refine_model_path.encode()
        opts[i].score_model_path = args.score_model_path.encode()
        opts[i].engine_cache_dir = args.engine_cache_dir.encode()

    start = time.perf_counter()
    group = lib.fp_group_create(opts, len(obj_ids), C.byref(config), err, len(err))
    assert group, f"fp_group_create failed: {err.value.decode()}"
    assert lib.fp_group_size(group) == len(obj_ids)
    rc = lib.fp_group_prepare(group, config.n_hypotheses, err, len(err))
    assert rc == 0, f"fp_group_prepare failed: {err.value.decode()}"
    print(f"created + prepared {len(obj_ids)} estimators in {time.perf_counter()-start:.2f}s")

    # Group register: one shared frame, per-object masks.
    masks = []
    for obj_id in obj_ids:
        rgb, depth, mask, k = contiguous_frame(layout, args.scene_id, args.image_id, obj_id, True)
        masks.append(mask)
    view = make_view(rgb, depth, k)
    mask_array = C.POINTER(C.c_uint8) * len(obj_ids)
    mask_ptrs = mask_array(*[m.ctypes.data_as(C.POINTER(C.c_uint8)) for m in masks])
    results = (rb.PoseResult * len(obj_ids))()
    statuses = (C.c_int * len(obj_ids))()

    start = time.perf_counter()
    rc = lib.fp_group_register_frame(group, C.byref(view), mask_ptrs, -1, -1,
                                     results, statuses, err, len(err))
    assert rc == 0, f"fp_group_register_frame failed ({rc}): {err.value.decode()}"
    print(f"group register of {len(obj_ids)} objects: {(time.perf_counter()-start)*1000:.0f} ms")
    assert all(s == 0 for s in statuses), f"unexpected statuses: {list(statuses)}"

    # Correctness: single-object register on the same estimator is deterministic
    # and must reproduce the group result exactly.
    for i, obj_id in enumerate(obj_ids):
        handle = C.c_void_p(lib.fp_group_handle(group, i))
        assert handle.value, f"fp_group_handle({i}) returned NULL"
        single_view = make_view(rgb, depth, k)
        single_view.mask_u8 = masks[i].ctypes.data_as(C.POINTER(C.c_uint8))
        single = rb.PoseResult()
        rc = lib.fp_register_frame(handle, C.byref(single_view), -1, -1, C.byref(single),
                                   err, len(err))
        assert rc == 0, f"fp_register_frame(obj {obj_id}) failed: {err.value.decode()}"
        delta = np.abs(pose_matrix(results[i]) - pose_matrix(single)).max()
        t_mm = pose_matrix(results[i])[:3, 3] * 1000
        print(f"  obj {obj_id:2d}: score={results[i].score:8.3f} "
              f"t=({t_mm[0]:8.2f},{t_mm[1]:8.2f},{t_mm[2]:8.2f}) mm "
              f"max|group-single|={delta:.2e}")
        assert delta < 1e-5, f"obj {obj_id}: group vs single register mismatch ({delta})"

    # Tracking: group (concurrent) vs serial single-handle calls, same frames.
    scene_camera = rb.read_json(layout.dataset_root / "test" / f"{args.scene_id:06d}"
                                / "scene_camera.json")
    next_ims = [i for i in sorted(int(x) for x in scene_camera)
                if i > args.image_id][:args.track_frames]
    assert next_ims, f"no frames after image {args.image_id} in scene {args.scene_id}"
    group_ms, serial_ms = [], []
    for im_id in next_ims:
        rgb2, depth2, _, k2 = contiguous_frame(layout, args.scene_id, im_id, obj_ids[0], False)
        view2 = make_view(rgb2, depth2, k2)

        start = time.perf_counter()
        rc = lib.fp_group_track_frame(group, C.byref(view2), None, -1, results, statuses,
                                      err, len(err))
        group_ms.append((time.perf_counter() - start) * 1000)
        assert rc == 0, f"fp_group_track_frame failed ({rc}): {err.value.decode()}"
        assert all(s == 0 for s in statuses), f"unexpected statuses: {list(statuses)}"

        start = time.perf_counter()
        for i in range(len(obj_ids)):
            handle = C.c_void_p(lib.fp_group_handle(group, i))
            single = rb.PoseResult()
            rc = lib.fp_track_frame(handle, C.byref(view2), -1, C.byref(single), err, len(err))
            assert rc == 0, f"serial fp_track_frame failed: {err.value.decode()}"
        serial_ms.append((time.perf_counter() - start) * 1000)
    print(f"track {len(obj_ids)} objects x {len(next_ims)} frames: "
          f"group median {np.median(group_ms):.2f} ms/frame, "
          f"serial median {np.median(serial_ms):.2f} ms/frame "
          f"({np.median(serial_ms)/np.median(group_ms):.2f}x)")

    # Per-object skip: NULL mask entries must be skipped without failing the call.
    skip_ptrs = mask_array(*[None if i > 0 else masks[0].ctypes.data_as(C.POINTER(C.c_uint8))
                             for i in range(len(obj_ids))])
    rc = lib.fp_group_register_frame(group, C.byref(view), skip_ptrs, -1, -1,
                                     results, statuses, err, len(err))
    assert rc == 0, f"skip register failed ({rc}): {err.value.decode()}"
    assert list(statuses) == [0] + [2] * (len(obj_ids) - 1), \
        f"expected OK + SKIPPED statuses, got {list(statuses)}"
    print(f"skip semantics ok: statuses={list(statuses)}")

    lib.fp_group_destroy(group)
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
