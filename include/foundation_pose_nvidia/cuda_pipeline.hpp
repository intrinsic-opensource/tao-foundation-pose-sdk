/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

#include "foundation_pose_nvidia/config.hpp"
#include "foundation_pose_nvidia/device_buffer.hpp"
#include "foundation_pose_nvidia/mesh.hpp"
#include "foundation_pose_nvidia/types.hpp"

namespace foundation_pose_nvidia {

struct DeviceMesh {
  int num_vertices = 0;
  int num_faces = 0;
  int texture_width = 0;
  int texture_height = 0;
  bool has_texture = false;
  bool has_uvs = false;

  DeviceBuffer vertices;       // float3[num_vertices], centered mesh frame.
  DeviceBuffer normals;        // float3[num_vertices].
  DeviceBuffer uvs;            // float2[num_vertices], optional.
  DeviceBuffer colors;         // float3[num_vertices], [0, 1].
  DeviceBuffer faces;          // int3[num_faces].
  DeviceBuffer texture_rgb;    // float3[texture_width * texture_height], optional.
};

struct DevicePoseResult {
  Mat4f pose;
  float score = 0.0f;
  int index = -1;
};

struct GpuWorkspace {
  explicit GpuWorkspace(const Config& config, int max_hypotheses);

  int max_width = 0;
  int max_height = 0;
  int max_pixels = 0;
  int max_hypotheses = 0;
  int crop_width = 0;
  int crop_height = 0;
  int crop_pixels = 0;
  int stats_blocks = 0;

  DeviceBuffer rgb_u8;
  DeviceBuffer mask_u8;
  DeviceBuffer depth_raw;
  DeviceBuffer depth_eroded;
  DeviceBuffer depth_filtered;
  DeviceBuffer xyz_map;

  DeviceBuffer block_stats;
  DeviceBuffer depth_histogram;
  DeviceBuffer init_translation;

  DeviceBuffer rotations;
  int rotation_count = 0;

  DeviceBuffer poses;
  DeviceBuffer previous_centered_pose;
  DeviceBuffer crop_boxes;

  DeviceBuffer rendered_rgb;
  DeviceBuffer rendered_xyz;
  DeviceBuffer network_rendered;
  DeviceBuffer network_observed;
  DeviceBuffer delta_translation;
  DeviceBuffer delta_rotation;
  DeviceBuffer scores;
  DeviceBuffer best_result;
};

void uploadMeshToDevice(const Mesh& mesh, DeviceMesh& dst, cudaStream_t stream);
void uploadRotationGridToDevice(const std::vector<Mat3f>& rotations,
                                GpuWorkspace& workspace,
                                cudaStream_t stream);

// Frame pointers may reference host, device, or managed memory (resolved via
// unified virtual addressing); device pointers must be usable on the
// estimator's device.
void uploadFrameToDevice(const std::uint8_t* rgb_u8,
                         const float* depth_m,
                         const std::uint8_t* mask_u8,
                         int width,
                         int height,
                         GpuWorkspace& workspace,
                         cudaStream_t stream);

void runDepthPreprocess(GpuWorkspace& workspace,
                        int width,
                        int height,
                        const Config& config,
                        cudaStream_t stream);

void runDepthToXyz(GpuWorkspace& workspace,
                   int width,
                   int height,
                   CameraIntrinsics intrinsics,
                   const Config& config,
                   cudaStream_t stream);

void runInitialTranslation(GpuWorkspace& workspace,
                           int width,
                           int height,
                           CameraIntrinsics intrinsics,
                           const Config& config,
                           cudaStream_t stream);

void seedPosesFromRotations(GpuWorkspace& workspace,
                            int batch_size,
                            cudaStream_t stream);

void computeCropBoxesOnDevice(const float* poses,
                              int batch_size,
                              CameraIntrinsics intrinsics,
                              float diameter,
                              const Config& config,
                              float* crop_boxes,
                              cudaStream_t stream);

void prepareNetworkInputsOnDevice(GpuWorkspace& workspace,
                                  const float* poses,
                                  int batch_size,
                                  int image_width,
                                  int image_height,
                                  float diameter,
                                  float invalid_depth_threshold,
                                  cudaStream_t stream);

void applyPoseDeltasOnDevice(float* poses,
                             const float* delta_translation,
                             const float* delta_rotation,
                             int batch_size,
                             float diameter,
                             const Config& config,
                             cudaStream_t stream);

void selectBestPoseOnDevice(const float* poses,
                            const float* scores,
                            int batch_size,
                            Vec3f mesh_center,
                            DevicePoseResult* result,
                            cudaStream_t stream);

void copyPoseOnDevice(const float* src_pose,
                      float* dst_pose,
                      cudaStream_t stream);

}  // namespace foundation_pose_nvidia
