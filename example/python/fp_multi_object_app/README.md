# YCB-V Multi-Object Registration

This example runs multi-object FoundationPose on one BOP/YCB-V scene using the
scene RGB image, depth image, visible masks, and PLY object models.

It supports either:

- a standard BOP `ycbv/` root with `test/` and `models/`, or
- the local layout in this workspace:
  - `foundation-pose/data/BOP_datasets/ycbv/test/<scene>/...`
  - `foundation-pose/data/BOP_datasets/ycbv/models/...`

## Dataset Download

From the `foundation-pose/` repo root, use the bundled helper script:

```bash
scripts/download_bop_ycbv.sh video
```

`video` downloads the full-frame `test_all` split needed for the app's default
tracking mode. If you only need first-frame registration, `scripts/download_bop_ycbv.sh all`
is enough because it fetches the smaller BOP19 test split instead.

The script downloads into `data/BOP_datasets/ycbv`, which produces the layout
the app expects:

```
foundation-pose/data/BOP_datasets/ycbv/
  models/               # obj_XXXXXX.ply + models_info.json
  models_eval/
  models_fine/
  test/000056/          # rgb/ depth/ mask_visib/ scene_camera.json scene_gt.json
  test_targets_bop19.json
  camera_cmu.json
  camera_uw.json
```

Pass `--models-dir` if your meshes live somewhere else; otherwise the app finds
the standard BOP `models/` directory automatically.

## Required Inputs

For the selected `--scene` and `--frame`, the app expects:

- `rgb/<frame>.png`
- `depth/<frame>.png`
- `mask_visib/<frame>_<gt_index>.png`
- `scene_camera.json`
- `scene_gt.json`
- `obj_XXXXXX.ply` models and `models_info.json`

The app uses the visible masks directly. It does not generate masks from GT
poses.

## Notes

- YCB-V meshes are loaded with `mesh_unit_scale=0.001`.
- Depth is converted with the per-frame BOP `depth_scale` from
  `scene_camera.json`.
- `--scene` is configurable so the same command works for any available YCB-V
  scene directory.
- `--objects` is optional and can filter by object id or object name, for
  example `2,4` or `obj_000002,obj_000004`.

## Build And Weights

Build the shared library and download the ONNX weights first:

```bash
CUDA_ARCHITECTURES=89 ./run_dev.sh run --rm build
scripts/download_weights.sh
```

That script installs the two files this app expects under `weights/`:
`weights/refiner_net.onnx` and `weights/score_net.onnx`.

## Default Run

From the `foundation-pose/` repo root:

```bash
example/python/fp_multi_object_app/run.sh \
  --dataset data/BOP_datasets/ycbv \
  --scene 000056 \
  --library build/libfoundation_pose_nvidia.so \
  --refine weights/refiner_net.onnx \
  --score weights/score_net.onnx \
  --cache engine_cache/ycbv_multi_object
```

With the defaults, the app:

- registers on frame `1`
- tracks every remaining frame in the same scene
- writes outputs to `example/python/fp_multi_object_app/results/`

To restrict registration to a subset of objects in the frame:

```bash
example/python/fp_multi_object_app/run.sh \
  --dataset data/BOP_datasets/ycbv \
  --scene 000056 \
  --frame 1 \
  --objects 2,4 \
  --library build/libfoundation_pose_nvidia.so \
  --refine weights/refiner_net.onnx \
  --score weights/score_net.onnx \
  --cache engine_cache/ycbv_multi_object
```

## Register Only

If you only want the first-frame registration result:

```bash
example/python/fp_multi_object_app/run.sh \
  --dataset data/BOP_datasets/ycbv \
  --scene 000056 \
  --frame 1 \
  --mode register \
  --library build/libfoundation_pose_nvidia.so \
  --refine weights/refiner_net.onnx \
  --score weights/score_net.onnx \
  --cache engine_cache/ycbv_multi_object
```

## Limited Tracking

To register on frame `1` and track only a smaller suffix of the scene:

```bash
example/python/fp_multi_object_app/run.sh \
  --dataset data/BOP_datasets/ycbv \
  --scene 000056 \
  --frame 1 \
  --track-frames 4 \
  --library build/libfoundation_pose_nvidia.so \
  --refine weights/refiner_net.onnx \
  --score weights/score_net.onnx \
  --cache engine_cache/ycbv_multi_object
```

## Outputs

By default, outputs are written to `example/python/fp_multi_object_app/results/`.
The directory contains:

- `poses.json`
- `inputs.json`
- `overlay.png`
- `rgb.png`
- `depth_m.npy`
- `depth_preview.png`
- one copied `mask_<object>.png` per registered object

`poses.json` uses the same schema in both modes: a `frames` list with one entry
per processed frame, each carrying its `phase` (`register` or `track`) and its
per-object estimates. In `--mode register` that list simply has one entry.

```json
{
  "frames": [
    {
      "frame": "000001",
      "phase": "register",
      "objects": [
        {
          "name": "obj_000002",
          "status_name": "OK",
          "score": -57.76,
          "pose_row_major": [],
          "gt_camera_from_object_opencv_row_major": [],
          "translation_error_m": 0.0030,
          "rotation_error_deg": 0.98
        }
      ]
    }
  ]
}
```

The `gt_*` and `*_error_*` fields appear only when scene GT poses are available
for that object. Note that `rotation_error_deg` is a plain geodesic angle and
does not account for object symmetry, so rotationally symmetric YCB-V objects
(for example `obj_000004`, a can) can report a large error while being correctly
aligned.
