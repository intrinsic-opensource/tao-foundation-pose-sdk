/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "foundation_pose_nvidia/config.hpp"
#include "foundation_pose_nvidia/types.hpp"

namespace foundation_pose_nvidia::detail {

std::vector<float> preprocessDepth(const DepthImage& depth, const Config& config);
std::vector<Vec3f> depthToXyz(const std::vector<float>& depth_m,
                              int width,
                              int height,
                              CameraIntrinsics intrinsics,
                              const Config& config);
Vec3f guessTranslation(const std::vector<float>& depth_m,
                       const MaskImage& mask,
                       CameraIntrinsics intrinsics,
                       const Config& config);
std::vector<Mat4f> generateHypotheses(const std::vector<float>& depth_m,
                                      const MaskImage& mask,
                                      CameraIntrinsics intrinsics,
                                      int n_hypotheses,
                                      const Config& config);
std::vector<Mat3f> makeRotationGrid(const Config& config);
std::vector<CropBox> computeCropBoxes(const std::vector<Mat4f>& poses,
                                      CameraIntrinsics intrinsics,
                                      float diameter,
                                      const Config& config);
Mat4f makeUncenter(Vec3f center);
Mat4f applyDelta(const Mat4f& pose,
                 Vec3f delta_translation,
                 Vec3f delta_rotation,
                 float diameter,
                 const Config& config);

}  // namespace foundation_pose_nvidia::detail
