/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include "foundation_pose_nvidia/cuda_pipeline.hpp"
#include "foundation_pose_nvidia/types.hpp"

namespace foundation_pose_nvidia {

class IRenderer {
 public:
  virtual ~IRenderer() = default;

  virtual void reserveForMesh(int num_vertices, int num_faces) {
    (void)num_vertices;
    (void)num_faces;
  }

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

}  // namespace foundation_pose_nvidia
