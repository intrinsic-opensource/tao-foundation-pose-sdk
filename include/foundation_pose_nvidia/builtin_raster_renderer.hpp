/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include <cuda_runtime_api.h>

#include "foundation_pose_nvidia/config.hpp"
#include "foundation_pose_nvidia/cuda_pipeline.hpp"
#include "foundation_pose_nvidia/device_buffer.hpp"
#include "foundation_pose_nvidia/renderer.hpp"
#include "foundation_pose_nvidia/types.hpp"

namespace foundation_pose_nvidia {

// Self-contained CUDA rasterizer with no third-party dependencies. Drop-in
// replacement for NvdiffrastRenderer: identical IRenderer contract and an
// identical intermediate rast-buffer format, so the shared shading and
// network-input packing consume its output unchanged.
//
// Algorithm: per-(hypothesis, triangle) atomic depth-test rasterization
// (64-bit z-buffer keyed by clip z/w with triangle-id tie-break), followed by
// a per-pixel resolve pass that recomputes perspective-correct barycentrics
// for the winning triangle. Sized for this pipeline's regime: small crops
// (default 160x160), meshes up to a few hundred thousand triangles, batches
// of up to ~252 pose hypotheses rendered in one call.
class BuiltinRasterRenderer final : public IRenderer {
 public:
  BuiltinRasterRenderer(const Config& config, int device_id, int max_batch);
  ~BuiltinRasterRenderer() override;

  BuiltinRasterRenderer(const BuiltinRasterRenderer&) = delete;
  BuiltinRasterRenderer& operator=(const BuiltinRasterRenderer&) = delete;

  void reserveForMesh(int num_vertices, int num_faces) override;

  void render(const DeviceMesh& mesh,
              const float* poses,
              const float* crop_boxes,
              int batch_size,
              CameraIntrinsics intrinsics,
              int image_width,
              int image_height,
              float3* out_rgb,
              float3* out_xyz,
              cudaStream_t stream) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace foundation_pose_nvidia
