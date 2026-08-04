/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <cuda_runtime_api.h>

#include "foundation_pose_nvidia/config.hpp"
#include "foundation_pose_nvidia/device_buffer.hpp"
#include "foundation_pose_nvidia/inference_runner.hpp"

namespace foundation_pose_nvidia {

class TensorRtRunner final : public IInferenceRunner {
 public:
  TensorRtRunner(RuntimeOptions options, Config config, int max_batch);
  ~TensorRtRunner() override;

  TensorRtRunner(const TensorRtRunner&) = delete;
  TensorRtRunner& operator=(const TensorRtRunner&) = delete;

  void prepareForBatch(int batch_size) override;

  void enqueueRefine(const float* rendered,
                     const float* observed,
                     float* delta_translation,
                     float* delta_rotation,
                     int batch_size,
                     cudaStream_t stream) override;

  void enqueueScore(const float* rendered,
                    const float* observed,
                    float* scores,
                    int batch_size,
                    cudaStream_t stream) override;

 private:
  class Engine;
  Engine& refineEngine();
  Engine& scoreEngine();

  RuntimeOptions options_;
  Config config_;
  int max_batch_ = 0;
  int engine_batch_ = 0;
  DeviceBuffer combined_refine_output_;
  std::unique_ptr<Engine> refine_;
  std::unique_ptr<Engine> score_;
};

}  // namespace foundation_pose_nvidia
