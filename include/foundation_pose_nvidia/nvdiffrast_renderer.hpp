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

class NvdiffrastRenderer final : public IRenderer {
 public:
  NvdiffrastRenderer(const Config& config, int device_id, int max_batch);
  ~NvdiffrastRenderer() override;

  NvdiffrastRenderer(const NvdiffrastRenderer&) = delete;
  NvdiffrastRenderer& operator=(const NvdiffrastRenderer&) = delete;

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
