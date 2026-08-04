# Examples

Runnable examples using the C API ([`cpp/`](cpp/)) and the Python bindings ([`python/`](python/)).

Build the library first (once):

```bash
./run_dev.sh run --rm build
```

## Single-object C++ example

Demonstrates the full register-then-track lifecycle across five RGB-D frames.

```bash
# BOP YCB-V mode (download data first):
scripts/download_bop_ycbv.sh
./run_dev.sh run --rm sample_app cpp

# Synthetic mode (no data needed):
./run_dev.sh run --rm sample_app cpp --use_synthesize
```

See [`cpp/README.md`](cpp/README.md) for output format and details.

## Single-object Python example

Registers once then tracks the same object across BOP YCB-V or synthetic RGB-D frames.

```bash
# BOP YCB-V mode (download data first):
scripts/download_bop_ycbv.sh
./run_dev.sh run --rm sample_app python

# Synthetic mode (no data needed):
./run_dev.sh run --rm sample_app python --use_synthesize
```

See [`python/fp_single_object_app/README.md`](python/fp_single_object_app/README.md) for flags and details.

## Multi-object Python example

Multi-object registration and tracking on a shared BOP/YCB-V scene frame.

```bash
./run_dev.sh run --rm sample_app python_multi \
  --dataset data/BOP_datasets/ycbv \
  --scene 000056 \
  --refine weights/refiner_net.onnx \
  --score weights/score_net.onnx \
  --cache engine_cache/ycbv_multi_object
```

See [`python/fp_multi_object_app/README.md`](python/fp_multi_object_app/README.md) for dataset download, object filtering, and tracking mode.

## Run both examples together

```bash
./run_dev.sh run --rm sample_app                           # C++ + Python (BOP mode)
./run_dev.sh run --rm sample_app cpp --use_synthesize      # C++ synthetic (no data needed)
./run_dev.sh run --rm sample_app python --use_synthesize   # Python synthetic
./run_dev.sh run --rm sample_app python_multi              # Python multi-object
```
