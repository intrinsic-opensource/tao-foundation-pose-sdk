# sample_app_cpp

A single-object FoundationPose C API example that demonstrates the full
register-then-track lifecycle across five RGB-D frames. Supports two modes
controlled by a command-line flag.

## Modes

### Default (BOP YCB-V real data)

Reads real RGB-D frames from BOP YCB-V scene 000048:

- Frame 1: register (uses `rgb/000001.png`, `depth/000001.png`, `mask_visib/000001_000000.png`)
- Frames 2-5: track (no mask; refines from the registered pose)

Camera intrinsics and depth scale are hardcoded for this scene
(`fx=1066.778`, `fy=1067.487`, `cx=312.987`, `cy=241.311`, depth_scale=0.1).

**Download the dataset first:**

```bash
bash scripts/download_bop_ycbv.sh   # downloads to data/BOP_datasets/
```

### Synthetic mode (`--use_synthesize`)

Generates RGB-D frames in code — no dataset required. A centered square at a
fixed depth provides the object signal, with a small per-frame shift to give
the tracker motion to follow.

## Build and run

Build the library first (once):

```bash
./run_dev.sh run --rm build
```

Then run the example:

```bash
# BOP YCB-V mode (download data first):
./run_dev.sh run --rm sample_app cpp

# Synthetic mode (no data needed):
./run_dev.sh run --rm sample_app cpp --use_synthesize

# Force a rebuild before running:
./run_dev.sh run --rm -e REBUILD=1 sample_app cpp
```

Or directly inside a container shell:

```bash
bash example/cpp/run.sh
bash example/cpp/run.sh --use_synthesize
```

## Output

Per-frame pose results printed to stdout (no CSV, no timing):

```
frame 1/5  register  score=  50.257  pose: [r00 r01 r02 tx / r10 r11 r12 ty / r20 r21 r22 tz]
frame 2/5  track     score=   1.000  pose: [...]
...
```

The pose is the 3x4 rotation+translation block of the 4x4 row-major SE(3)
object-to-camera transform (translation in meters).
