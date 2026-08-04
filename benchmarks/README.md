# Benchmark (BOP YCB-V)

## Dataset

```bash
scripts/download_bop_ycbv.sh          # base + models + toolkit (register / one-shot AR)
scripts/download_bop_ycbv.sh video    # + the full-frame test split for tracking (~15 GB)
scripts/download_weights.sh           # ONNX weights into $FP_WEIGHTS_DIR
```

Expected layout under `$FP_DATA_DIR`:

```
BOP_datasets/
  ycbv/           # models/, test/, test_targets_bop19.json, ...
  bop_toolkit/    # github.com/thodan/bop_toolkit
```

The default `test` split ships the **sparse** BOP19 keyframes (75/scene) — right for
register / one-shot AR. The `video` preset (`test_all`) adds **every consecutive video
frame** (20,738 across the 12 test scenes), which is what frame-to-frame tracking needs.

## Build the benchmark image

```bash
./run_dev.sh build bench
```

## Register + BOP19 evaluation (one-shot)

```bash
./run_dev.sh run --rm bench --benchmark --mode register --evaluate \
  --bop-toolkit-path /work/data/BOP_datasets/bop_toolkit \
  --eval-renderer-type vispy --eval-num-workers 8 --cleanup-eval \
  --n-refine 5 --n-hypotheses 252 --prepare-estimators --group-register-by-object \
  --checkpoint-interval 100 --max-register-estimators 1 \
  --output_dir ./benchmarks/results/register
```

## Video (dense) tracking + BOP19 evaluation

Register the first frame of each sequence, then track **every consecutive frame**
(`--track-all-frames`, needs the `video` / `test_all` split). Poses are scored on the
same BOP19 keyframes as register, so the AR is directly comparable.

```bash
./run_dev.sh run --rm bench --benchmark --mode tracking --track-all-frames --evaluate \
  --bop-toolkit-path /work/data/BOP_datasets/bop_toolkit \
  --eval-renderer-type vispy --eval-num-workers 8 --cleanup-eval \
  --n-refine 5 --n-track-refine 2 --n-hypotheses 252 --prepare-estimators \
  --group-register-by-object --max-register-estimators 1 \
  --output_dir ./benchmarks/results/dense_tracking
```

Useful variants:

- **Precision:** `--precision fp32|tf32|fp16|bf16` (default `tf32`; use `fp32` for
  strict IEEE FP32 — best cross-GPU accuracy parity, e.g. x86 vs Jetson). Engines are
  cached per precision.
- **Sparse tracking:** drop `--track-all-frames` to track only the BOP19 keyframes (no
  `video` split needed).
- **`--mode both`** runs register then tracking.
- **Quick subset:** `--scene-ids 48` (single scene) or `--max-targets 75`; smallest
  smoke run: `./run_dev.sh run --rm bench --benchmark --mode register --max-targets 6`.
- **Apple-to-apple:** add `--no-use-symmetry-hypothesis-counts`.

Outputs land in `--output_dir`: `foundationpose_ycbv-test.csv`, timing JSON, and
`eval_official/.../scores_bop19.json` (BOP19 AR). Run
`./run_dev.sh run --rm bench --help` for the full option list.

## Reference results

Full YCB-V (4,123 keyframe targets, `--n-refine 5`, `--n-hypotheses 252`), built-in
CUDA rasterizer. Latency in milliseconds (ms), mean per target/frame (`by_type` latency
from timing JSON — register uses `by_type["register"]`, tracking uses `by_type["track"]`).
RTX PRO 6000 figures are measured on the **RTX PRO 6000 Blackwell Max-Q Workstation Edition**.

The two recommended operating points are **TF32** (the default — full accuracy parity,
see below) and **FP16** (fastest); their rows are **bold** in the latency tables.

### Register mode — latency

| Precision | RTX PRO 6000 Blackwell — (ms) | Jetson AGX Thor — (ms) |
| --------- | ----------------------------- | ---------------------- |
| FP32      | 703.18                        | 6192.22                |
| **TF32**  | **323.98**                    | **854.63**             |
| BF16      | 154.13                        | 447.48                 |
| **FP16**  | **148.37**                    | **425.75**             |

### Tracking mode — latency (per frame)

| Precision | RTX PRO 6000 Blackwell — (ms) | Jetson AGX Thor — (ms) |
| --------- | ----------------------------- | ---------------------- |
| FP32      | 12.34                         | 18.96                  |
| **TF32**  | **2.73**                      | **10.64**              |
| BF16      | 2.66                          | 7.57                   |
| **FP16**  | **2.01**                      | **7.17**               |

### Accuracy — BOP19 Average Recall (AR)

Register is one-shot per keyframe; **video tracking** registers the first frame of each
sequence then tracks every consecutive frame (`test_all`, gap = 1) — the meaningful
steady-state metric, and register-grade.

**RTX PRO 6000 Blackwell**

| Precision | Register | Video tracking |
| --------- | -------- | -------------- |
| FP32      | 0.9159   | 0.9049         |
| TF32      | 0.9156   | 0.9053         |
| BF16      | 0.9131   | 0.9128         |
| FP16      | 0.9165   | 0.9054         |

_Jetson AGX Thor gives near-identical accuracy — AR within ~0.002 of RTX PRO 6000 at
every precision._

How the accuracy is measured — the two columns use different commands. Sweep
`--precision tf32|fp16` for each row; the AR lands in `scores_bop19.json`:

```bash
# Register column (one-shot per keyframe):
./run_dev.sh run --rm bench --benchmark --mode register --evaluate \
  --bop-toolkit-path /work/data/BOP_datasets/bop_toolkit \
  --eval-renderer-type vispy --eval-num-workers 8 --cleanup-eval \
  --n-refine 5 --n-hypotheses 252 --prepare-estimators --group-register-by-object \
  --checkpoint-interval 100 --max-register-estimators 1 --precision tf32 \
  --output_dir ./benchmarks/results/register

# Video-tracking column (register frame 1, then track every consecutive frame):
./run_dev.sh run --rm bench --benchmark --mode tracking --track-all-frames --evaluate \
  --bop-toolkit-path /work/data/BOP_datasets/bop_toolkit \
  --eval-renderer-type vispy --eval-num-workers 8 --cleanup-eval \
  --n-refine 5 --n-track-refine 2 --n-hypotheses 252 --prepare-estimators \
  --group-register-by-object --max-register-estimators 1 --precision tf32 \
  --output_dir ./benchmarks/results/dense_tracking

# AR for either run:
#   cat <output_dir>/eval_official/*/*/scores_bop19.json
```

Video tracking is within ~0.01 AR of one-shot register across all precisions.
