# Python Examples

Runnable examples for the installable [`foundation_pose_nvidia`](../../python/)
package.

## Single Object

[`fp_single_object_app/`](fp_single_object_app/) shows the high-level Python API with one
estimator. It registers once and then tracks the same object across BOP YCB-V
or synthetic RGB-D frames.

```bash
./run_dev.sh run --rm sample_app python
./run_dev.sh run --rm sample_app python --use_synthesize
```

## Multi Object

[`fp_multi_object_app/`](fp_multi_object_app/) shows a shared-frame, multi-object registration
flow using one BOP/YCB-V scene. The app loads RGB-D, `mask_visib`, intrinsics,
and object models from the selected scene and then uses the multi-object group
runner from the Python package.

```bash
./run_dev.sh run --rm sample_app python_multi \
  --dataset data/BOP_datasets/ycbv \
  --scene 000056 \
  --refine weights/refiner_net.onnx \
  --score weights/score_net.onnx \
  --cache engine_cache/ycbv_multi_object
```
