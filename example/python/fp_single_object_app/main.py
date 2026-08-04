#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-object FoundationPose Python API with synthetic or BOP RGB-D.

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from foundation_pose_nvidia import Estimator, EstimatorOptions, RgbdFrame, RuntimeConfig, load_library

NUMBER_OF_FRAMES = 5
WIDTH = 640
HEIGHT = 480
# FP_N_HYPOTHESES overrides the default (252) to shrink GPU memory use.
N_HYPOTHESES = int(os.environ.get("FP_N_HYPOTHESES", "252"))
MESH_UNIT_SCALE = 0.001
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# BOP YCB-V dataset constants
BOP_SCENE_DIR = Path("data/BOP_datasets/ycbv/test/000048")
BOP_OBJ_ID = 1  # object we track in scene 000048


@dataclass
class Record:
    frame_idx: int
    phase: str
    ok: bool
    score: float
    time_s: float
    pose: np.ndarray


def _find_gt_id(scene_gt: dict, im_id: int, obj_id: int) -> int:
    for idx, entry in enumerate(scene_gt.get(str(im_id), [])):
        if entry["obj_id"] == obj_id:
            return idx
    raise ValueError(f"obj_id={obj_id} not found in scene_gt for im_id={im_id}")


def load_bop_frame(
    bop_dir: Path,
    im_id: int,
    cam: dict,
    gt_id: int | None,
    need_mask: bool,
) -> RgbdFrame:
    """Load a single RGB-D frame from a BOP scene directory."""
    fid = f"{im_id:06d}"
    depth_scale = float(cam.get("depth_scale", 1.0))
    k = np.asarray(cam["cam_K"], dtype=np.float32).reshape(3, 3)

    bgr = cv2.imread(str(bop_dir / "rgb" / f"{fid}.png"))
    if bgr is None:
        raise FileNotFoundError(f"RGB not found: {bop_dir}/rgb/{fid}.png")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth_raw = cv2.imread(str(bop_dir / "depth" / f"{fid}.png"), cv2.IMREAD_ANYDEPTH)
    if depth_raw is None:
        raise FileNotFoundError(f"Depth not found: {bop_dir}/depth/{fid}.png")
    depth_m = depth_raw.astype(np.float32) * depth_scale / 1000.0

    mask = None
    if need_mask:
        if gt_id is None:
            raise ValueError(f"gt_id required for mask but not found for im_id={im_id}")
        mask_path = bop_dir / "mask_visib" / f"{fid}_{gt_id:06d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

    return RgbdFrame(rgb=rgb, depth_m=depth_m, intrinsics=k, mask=mask)


def fill_frame(width: int, height: int, frame_index: int) -> RgbdFrame:
    rgb = np.full((height, width, 3), 96, dtype=np.uint8)
    depth_m = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    intrinsics = np.array(
        [[float(width), 0.0, width * 0.5], [0.0, float(width), height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    box = min(width, height) // 3
    shift = frame_index * 2
    x0 = (width - box) // 2 + shift
    y0 = (height - box) // 2
    y1 = min(y0 + box, height)
    x1 = min(x0 + box, width)
    rgb[y0:y1, x0:x1] = (180, 130, 90)
    depth_m[y0:y1, x0:x1] = 0.7
    mask[y0:y1, x0:x1] = 255
    return RgbdFrame(rgb=rgb, depth_m=depth_m, intrinsics=intrinsics, mask=mask)


def write_csv(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "phase", "ok", "score", "time_s", "pose_row_major"])
        for rec in records:
            pose_flat = " ".join(f"{v:.6f}" for v in rec.pose.reshape(-1))
            writer.writerow([rec.frame_idx, rec.phase, int(rec.ok), rec.score, rec.time_s, pose_flat])
    print(f"\nwrote {len(records)} records to {path}")


def print_timing_summary(records: list[Record]) -> None:
    reg = [r.time_s for r in records if r.ok and r.phase == "register"]
    trk = [r.time_s for r in records if r.ok and r.phase == "track"]
    print("\n=== timing summary (steady-state, per call) ===")
    if reg:
        print(f"  register: n={len(reg)}  mean={1e3 * sum(reg) / len(reg):.1f} ms  "
              f"max={1e3 * max(reg):.1f} ms")
    if trk:
        print(f"  track   : n={len(trk)}  mean={1e3 * sum(trk) / len(trk):.1f} ms  "
              f"max={1e3 * max(trk):.1f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cad",
        default=os.environ.get(
            "FP_CAD_PATH",
            "data/BOP_datasets/ycbv/models/obj_000001.ply",
        ),
        help="CAD mesh path (OBJ/PLY).",
    )
    parser.add_argument(
        "--use_synthesize",
        action="store_true",
        help="Use synthetic RGB-D instead of BOP dataset frames.",
    )
    parser.add_argument(
        "--library",
        default=os.environ.get("FP_LIBRARY"),
        help="Path to libfoundation_pose_nvidia.so (default: auto-discover).",
    )
    parser.add_argument(
        "--refine-model-path",
        default=os.environ.get("FP_REFINE_MODEL_PATH"),
        help="Path to refiner_net.onnx.",
    )
    parser.add_argument(
        "--score-model-path",
        default=os.environ.get("FP_SCORE_MODEL_PATH"),
        help="Path to score_net.onnx.",
    )
    parser.add_argument(
        "--engine-cache-dir",
        default=os.environ.get("FP_ENGINE_CACHE_DIR", "engine_cache"),
        help="TensorRT engine cache directory.",
    )
    parser.add_argument("--mesh-unit-scale", type=float, default=MESH_UNIT_SCALE)
    parser.add_argument("--frames", type=int, default=NUMBER_OF_FRAMES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.refine_model_path or not args.score_model_path:
        print("error: set --refine-model-path and --score-model-path or FP_* env vars", file=sys.stderr)
        return 2

    # Validate BOP data directory when not using synthetic mode
    if not args.use_synthesize:
        if not BOP_SCENE_DIR.exists():
            print(
                f"error: BOP data not found at {BOP_SCENE_DIR}/\n"
                "Run:  scripts/download_bop_ycbv.sh  (downloads to data/BOP_datasets/)",
                file=sys.stderr,
            )
            sys.exit(2)

    options = EstimatorOptions.from_env(
        args.cad,
        refine_model_path=args.refine_model_path,
        score_model_path=args.score_model_path,
        engine_cache_dir=args.engine_cache_dir,
        mesh_unit_scale=args.mesh_unit_scale,
    )
    config = RuntimeConfig(
        max_image_width=WIDTH,
        max_image_height=HEIGHT,
        n_hypotheses=N_HYPOTHESES,
    )

    records: list[Record] = []
    with Estimator(
        options,
        config,
        prepare_batch=N_HYPOTHESES,
    ) as est:
        if args.use_synthesize:
            # Synthetic mode: use fill_frame for --frames frames
            total_frames = args.frames
            for frame_idx in range(total_frames):
                frame = fill_frame(WIDTH, HEIGHT, frame_idx)
                is_register = frame_idx == 0
                phase = "register" if is_register else "track"
                try:
                    if is_register:
                        result = est.register(frame)
                    else:
                        frame.mask = None
                        result = est.track(frame)
                except Exception as exc:
                    print(f"frame {frame_idx + 1}/{total_frames}  {phase:8s}  FAILED: {exc}")
                    records.append(
                        Record(frame_idx, phase, False, 0.0, 0.0, np.eye(4, dtype=np.float32))
                    )
                    if is_register:
                        break
                    continue

                t = result.pose[:3, 3]
                print(
                    f"frame {frame_idx + 1}/{total_frames}  {phase:8s}  score={result.score:8.3f}  "
                    f"t=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m  {1e3 * result.elapsed_s:.1f} ms"
                )
                records.append(
                    Record(frame_idx, phase, True, result.score, result.elapsed_s, result.pose)
                )
        else:
            # BOP mode: read real frames; image IDs come from scene_camera.json keys.
            scene_camera = json.loads((BOP_SCENE_DIR / "scene_camera.json").read_text())
            scene_gt = json.loads((BOP_SCENE_DIR / "scene_gt.json").read_text())
            bop_frame_ids = sorted(int(k) for k in scene_camera)[:args.frames]
            total_frames = len(bop_frame_ids)
            for frame_idx, im_id in enumerate(bop_frame_ids):
                is_register = frame_idx == 0
                phase = "register" if is_register else "track"
                cam = scene_camera[str(im_id)]
                gt_id = _find_gt_id(scene_gt, im_id, BOP_OBJ_ID) if is_register else None
                try:
                    frame = load_bop_frame(BOP_SCENE_DIR, im_id, cam, gt_id, need_mask=is_register)
                    if is_register:
                        result = est.register(frame)
                    else:
                        result = est.track(frame)
                except Exception as exc:
                    print(f"frame {frame_idx + 1}/{total_frames}  {phase:8s}  FAILED: {exc}")
                    records.append(
                        Record(frame_idx, phase, False, 0.0, 0.0, np.eye(4, dtype=np.float32))
                    )
                    if is_register:
                        break
                    continue

                t = result.pose[:3, 3]
                print(
                    f"frame {frame_idx + 1}/{total_frames}  {phase:8s}  score={result.score:8.3f}  "
                    f"t=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m  {1e3 * result.elapsed_s:.1f} ms"
                )
                records.append(
                    Record(frame_idx, phase, True, result.score, result.elapsed_s, result.pose)
                )

    write_csv(OUTPUT_DIR / "single_object_results.csv", records)
    print_timing_summary(records)
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
