/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

namespace foundation_pose_nvidia {

class IInferenceRunner {
 public:
  virtual ~IInferenceRunner() = default;

  virtual void prepareForBatch(int batch_size) { (void)batch_size; }

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

}  // namespace foundation_pose_nvidia
