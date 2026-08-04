# Architecture

This implementation keeps the FoundationPose algorithm in a small C++ core and
separates the two vendor-heavy pieces, rendering and neural network inference,
behind interfaces. The default build uses a built-in CUDA rasterizer and
TensorRT for ONNX execution, but the central `FoundationPose` class is not tied
to those concrete classes.

## Goals

- Keep register and track execution mostly on the GPU.
- Avoid Python, PyTorch, or CPU fallback in the production path.
- Reuse expensive resources across frames: TensorRT engines, CUDA buffers,
  uploaded meshes, and renderer state.
- Make renderer and inference backends replaceable without rewriting pose logic.
- Preserve a simple C ABI for non-C++ callers.

## Data Flow

### Register Mode

The one-shot global pose search, run on first sight of an object (or to
re-acquire it after tracking loss). Requires an object mask from the caller's
detector/segmenter.

```
 INPUT
   RGB (HWC uint8) · Depth (HW float32, meters) · Mask (HW uint8) · K (3×3 intrinsics)
               │
               ▼
   ┌───────────────────────┐
   │ depth preprocessing   │
   └───────────┬───────────┘
               ▼
   ┌───────────────────────┐
   │ generate hypotheses   │
   └───────────┬───────────┘
               ▼
       ┌───────────────────────────────────────────────────┐
       │                × refine iterations                │
       │   ┌─────────────────────┐    ┌────────────────┐   │
       │   │ render hypotheses   │ ◄──┤ IRenderer      │   │
       │   └──────────┬──────────┘    │ (rasterizer)   │   │
       │              ▼               └────────────────┘   │
       │   ┌─────────────────────┐    ┌────────────────┐   │
       │   │ refine poses        │ ◄──┤ RefineNet      │   │
       │   └──────────┬──────────┘    │ (TensorRT)     │   │
       │              │               └────────────────┘   │
       └──────────────┼────────────────────────────────────┘
                      ▼
   ┌───────────────────────┐          ┌────────────────┐
   │ score & select best   │ ◄────────┤ ScoreNet       │
   └───────────┬───────────┘          │ (TensorRT)     │
               ▼                      └────────────────┘
 OUTPUT
   pose  4×4 row-major object-to-camera SE(3), translation in meters
   score confidence of the best hypothesis      (the pose also seeds tracking)
```

### Track Mode

Per-frame refinement; no mask. The previous pose is kept inside the estimator
and locates the object in the new frame — the refined pose feeds back as the
seed for the next frame.

```
 INPUT
   RGB (HWC uint8) · Depth (HW float32, meters) · K (3×3 intrinsics)
               │
               ▼
   ┌───────────────────────┐
   │ depth preprocessing   │
   └───────────┬───────────┘
               ▼                        previous pose (kept on GPU)
   ┌───────────────────────┐                         │
   │ crop from prev pose   │ ◄───────────────────────┤
   └───────────┬───────────┘                         │
               ▼                                     │
   ┌───────────────────────┐    ┌────────────────┐   │
   │ render at prev pose   │ ◄──┤ IRenderer      │   │
   └───────────┬───────────┘    │ (rasterizer)   │   │
               ▼                └────────────────┘   │
   ┌───────────────────────┐    ┌────────────────┐   │
   │ refine pose (× iters) │ ◄──┤ RefineNet      │   │
   └───────────┬───────────┘    │ (TensorRT)     │   │
               │                └────────────────┘   │
               ├───────── feedback: seeds next frame ┘
               ▼
 OUTPUT
   pose  4×4 object-to-camera SE(3) · score 1.0
```

Track mode runs a single pose, while register mode refines and scores many
hypotheses in one batch.

## Integration Workflows

### C ABI

```c
fp_default_config(&cfg);                    // recommended defaults
cfg.tensorrt_precision = FP_PRECISION_FP32; // optional: TF32 default; FP32/FP16/BF16
h = fp_create(&opts, &cfg, err, sizeof err);   // CAD mesh (or fp_create_model_free)
fp_prepare(h, cfg.n_hypotheses, ...);       // pre-build TensorRT contexts (optional)
fp_register_frame(h, &frame, -1, -1, &pose, ...);  // first frame: needs mask
while (streaming)
    fp_track_frame(h, &frame, -1, &pose, ...);     // every frame: no mask
fp_destroy(h);
```

For **multiple objects** on one shared frame, the `fp_group_*` API owns one
estimator per object and evaluates them concurrently (one CUDA stream each,
TensorRT engines shared across objects):
`fp_group_create` → `fp_group_register_frame(frame, masks[])` →
`fp_group_track_frame(frame)` → `fp_group_destroy`. See the Doxygen reference
in `c_api.h`.

### Python

```python
from foundation_pose_nvidia import Estimator, EstimatorOptions, RgbdFrame, RuntimeConfig

options = EstimatorOptions.from_env("/path/to/mesh.ply")
with Estimator(options, RuntimeConfig()) as est:
    result = est.register(RgbdFrame(rgb, depth_m, K, mask))   # first frame
    result = est.track(RgbdFrame(rgb, depth_m, K))            # every frame
    # result.pose (4x4), result.score, result.elapsed_s
```

`RgbdFrame` accepts numpy arrays or GPU arrays (PyTorch CUDA tensors, CuPy,
Numba) per buffer — see `python/README.md`.

### Custom renderer / inference backend (C++ only)

The C ABI and Python bindings use the built-in CUDA rasterizer by default
(`FP_RENDERER=nvdiffrast` selects the optional nvdiffrast plugin) and TensorRT. Substituting your own renderer or inference engine means implementing
`IRenderer` / `IInferenceRunner` and injecting it through the C++ constructor —
see [Swapping In a New Renderer](#swapping-in-a-new-renderer).

## Key Decisions

### 1. Interfaces Around Renderer and Inference

The estimator owns algorithmic sequencing, but it does not know how rendering or
inference are implemented. It only depends on:

```cpp
class IRenderer {
 public:
  virtual ~IRenderer() = default;
  virtual void reserveForMesh(int num_vertices, int num_faces);
  virtual void render(const DeviceMesh& mesh,
                      const float* poses,
                      const float* crop_boxes,
                      int batch_size,
                      CameraIntrinsics intrinsics,
                      int image_width,
                      int image_height,
                      float3* out_rgb,
                      float3* out_xyz,
                      cudaStream_t stream) = 0;
};
```

and:

```cpp
class IInferenceRunner {
 public:
  virtual ~IInferenceRunner() = default;
  virtual void prepareForBatch(int batch_size);
  virtual void enqueueRefine(const float* rendered,
                             const float* observed,
                             float* delta_translation,
                             float* delta_rotation,
                             int batch_size,
                             cudaStream_t stream) = 0;
  virtual void enqueueScore(const float* rendered,
                            const float* observed,
                            float* scores,
                            int batch_size,
                            cudaStream_t stream) = 0;
};
```

This is the core composability decision. A new renderer can be introduced
without changing depth preprocessing, crop generation, TensorRT binding, scoring,
tracking state, or the C ABI.

### 2. Renderer Output Contract Is Tensor-Oriented

`IRenderer::render` receives already-uploaded mesh data, device poses, device
crop boxes, camera intrinsics, and the target CUDA stream. It writes two
device-resident NHWC crop buffers:

- `out_rgb`: `batch_size * input_height * input_width` `float3` colors in
  normalized RGB space.
- `out_xyz`: `batch_size * input_height * input_width` `float3` camera-space XYZ.

The renderer does not prepare TensorRT inputs directly. That stays in
`prepareNetworkInputsOnDevice`, which combines rendered crops with observed
RGB-D crops and writes NCHW 6-channel tensors. This boundary keeps renderers
focused on geometry and visibility, not model-specific layout.

### 3. Frame Input May Be Host or Device Memory

`fp_image_view_t` buffers (RGB/depth/mask) accept host, CUDA device, or managed
pointers. Ingest copies use `cudaMemcpyDefault`, which resolves the source
location via unified virtual addressing — device input becomes a
device-to-device copy into the workspace. The Python `RgbdFrame` maps GPU arrays
(anything exposing `__cuda_array_interface__`, e.g. torch CUDA tensors) straight
onto this path. Intrinsics and model-free reference views remain host-only.

## Swapping In a New Renderer

A renderer replacement only needs to satisfy `IRenderer`. The estimator will
still handle:

- Mesh upload into `DeviceMesh`.
- Frame upload and depth preprocessing.
- Pose hypothesis generation.
- Crop box computation.
- Network input packing.
- Refiner/scorer execution.
- Best-pose selection.

### Minimal Skeleton

```cpp
#include "foundation_pose_nvidia/renderer.hpp"

class MyRenderer final : public foundation_pose_nvidia::IRenderer {
 public:
  MyRenderer(const foundation_pose_nvidia::Config& config, int device_id, int max_batch);

  void reserveForMesh(int num_vertices, int num_faces) override {
    // Optional: allocate renderer-specific device buffers here.
  }

  void render(const foundation_pose_nvidia::DeviceMesh& mesh,
              const float* poses,
              const float* crop_boxes,
              int batch_size,
              foundation_pose_nvidia::CameraIntrinsics intrinsics,
              int image_width,
              int image_height,
              float3* out_rgb,
              float3* out_xyz,
              cudaStream_t stream) override {
    // Launch CUDA work that fills out_rgb and out_xyz for every crop.
    // The caller owns synchronization through the provided stream.
  }
};
```

Then inject it through the C++ constructor:

```cpp
using namespace foundation_pose_nvidia;

RuntimeOptions runtime;
runtime.refine_model_path = "/models/refiner_net.onnx";
runtime.score_model_path = "/models/score_net.onnx";
runtime.engine_cache_dir = "engine_cache";

Config config;
auto inference = std::make_unique<TensorRtRunner>(runtime, config, config.n_hypotheses);
auto renderer = std::make_unique<MyRenderer>(config, runtime.device_id, config.n_hypotheses);

Mesh mesh = loadMesh("/data/ycbv/models/obj_000001.ply");
PreprocessedMesh preprocessed = preprocessMesh(mesh, config);

FoundationPose estimator(
    std::move(preprocessed),
    runtime,
    std::move(inference),
    std::move(renderer),
    config);
```

### Example: OpenGL or Vulkan Renderer

For an OpenGL or Vulkan renderer, the practical approach is:

1. Keep `DeviceMesh` as the common mesh description.
2. In `reserveForMesh`, import or copy mesh buffers into the graphics API's
   preferred buffers.
3. In `render`, render one crop per hypothesis, then expose/copy the resulting
   color and XYZ images into the CUDA pointers supplied by `out_rgb` and
   `out_xyz`.
4. Respect the provided CUDA stream. If the graphics API uses external
   semaphores, signal/wait so downstream CUDA kernels see completed output.

The estimator does not care whether those pixels came from nvdiffrast, Vulkan,
OpenGL interop, OptiX, or a custom CUDA rasterizer as long as the output buffers
match the `IRenderer` contract.

### Renderer Invariants

A replacement renderer must preserve these invariants:

- All output pointers are device pointers.
- Output dimensions are `batch_size x config.input_height x config.input_width`.
- Output layout is NHWC, one `float3` per pixel.
- RGB values are normalized floats in `[0, 1]`.
- XYZ values are camera-space meters. Invalid/background pixels should have a
  depth below the invalid threshold used by `prepareNetworkInputs`.
- Work is enqueued on the supplied CUDA stream or synchronized with it.
- The renderer must not free or retain ownership of `DeviceMesh` buffers.

A custom **inference backend** follows the same pattern: implement
`IInferenceRunner` (see the interface above) and inject it through the same
constructor. It receives packed NCHW `batch_size × 6 × H × W` rendered/observed
tensors and writes per-pose translation/rotation deltas and scores on the
provided CUDA stream.
