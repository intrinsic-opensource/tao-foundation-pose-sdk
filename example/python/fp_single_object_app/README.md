# Single-Object Python API With BOP or Synthetic RGB-D Frames

Runnable Python example using the installable [`foundation_pose_nvidia`](../../../python/)
package (`bindings` + `core` layers, NVIDIA cuda-python style).

The app supports two modes:

- **BOP mode** (default): reads real RGB-D frames from the YCB-V BOP dataset (scene 48).
- **Synthetic mode** (`--use_synthesize`): generates simple synthetic RGB-D frames (no dataset required).

## Install

The compose services install the package automatically (`pip install -e python/`).
Manual install inside the container:

```bash
pip install -e python/
```

## Run — BOP mode (default)

Download the YCB-V BOP dataset first (≈ 5 GB):

```bash
bash scripts/download_bop_ycbv.sh   # downloads to data/BOP_datasets/
```

Then build the library and run the app:

```bash
./run_dev.sh run --rm build          # once: builds build/libfoundation_pose_nvidia.so
./run_dev.sh run --rm sample_app python  # registers on frame 1, tracks frames 2-5
```

BOP mode registers the object on frame 1 and tracks it on frames 2–5 using the
YCB-V scene 48 images (`data/BOP_datasets/ycbv/test/000048/`).

## Run — synthetic mode

Pass `--use_synthesize` to skip the BOP dataset and generate simple RGB-D frames
in memory instead:

```bash
./run_dev.sh run --rm sample_app python --use_synthesize
./run_dev.sh run --rm sample_app python --use_synthesize --frames 10
```

## Optional flags

```bash
./run_dev.sh run --rm sample_app python \
  --cad data/BOP_datasets/ycbv/models/obj_000001.ply \
  --mesh-unit-scale 0.001 \
  --frames 5 \
  --use_synthesize
```

| Flag | Default | Description |
|---|---|---|
| `--use_synthesize` | off | Use synthetic RGB-D instead of BOP dataset frames |
| `--cad` | `$FP_CAD_PATH` or `data/BOP_datasets/ycbv/models/obj_000001.ply` | CAD mesh path (OBJ/PLY) |
| `--mesh-unit-scale` | `0.001` | Scale factor from mesh units to metres |
| `--frames` | `5` | Number of frames (synthetic mode only) |
| `--library` | `$FP_LIBRARY` | Path to `libfoundation_pose_nvidia.so` |
| `--refine-model-path` | `$FP_REFINE_MODEL_PATH` | Path to `refiner_net.onnx` |
| `--score-model-path` | `$FP_SCORE_MODEL_PATH` | Path to `score_net.onnx` |
| `--engine-cache-dir` | `$FP_ENGINE_CACHE_DIR` or `engine_cache` | TensorRT engine cache directory |

Model/cache paths default to the `FP_*` env vars set in `compose.yaml` / `.env`, so you
normally don't need to pass them.

To run inside an interactive shell instead:

```bash
./run_dev.sh up -d dev
./run_dev.sh exec dev bash example/python/fp_single_object_app/run.sh
./run_dev.sh down
```

## Output

Per-frame pose/score/timing prints to the console and is written to
`results/single_object_results.csv`.

See [`python/README.md`](../../../python/README.md) for API documentation.
