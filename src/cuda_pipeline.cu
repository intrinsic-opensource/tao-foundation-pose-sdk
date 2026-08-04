/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "foundation_pose_nvidia/cuda_pipeline.hpp"
#include "foundation_pose_nvidia/exception.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace foundation_pose_nvidia {
namespace {

constexpr int kBlockSize = 256;
constexpr int kDepthHistogramBins = 65536;

struct TranslationStats {
  int min_x;
  int max_x;
  int min_y;
  int max_y;
  int count;
  float depth_sum;
};

__device__ bool validDepth(float z, const Config config) {
  return isfinite(z) && z >= config.depth_min && z < config.depth_max;
}

__device__ float3 mat4TransformPoint(const float* m, float3 p) {
  return make_float3(m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
                     m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
                     m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
}

__device__ float3 mat3Mul(const float* m, float3 p) {
  return make_float3(m[0] * p.x + m[1] * p.y + m[2] * p.z,
                     m[3] * p.x + m[4] * p.y + m[5] * p.z,
                     m[6] * p.x + m[7] * p.y + m[8] * p.z);
}

__device__ void so3ExpTranspose(float3 omega, float* out) {
  const float theta = sqrtf(fmaxf(omega.x * omega.x + omega.y * omega.y +
                                      omega.z * omega.z,
                                  0.0f));
  for (int i = 0; i < 9; ++i) {
    out[i] = 0.0f;
  }
  out[0] = out[4] = out[8] = 1.0f;
  if (theta < 1e-6f) {
    return;
  }
  const float inv_theta = 1.0f / theta;
  const float x = omega.x * inv_theta;
  const float y = omega.y * inv_theta;
  const float z = omega.z * inv_theta;
  const float s = sinf(theta);
  const float c = cosf(theta);
  const float one_c = 1.0f - c;

  float r[9];
  r[0] = c + x * x * one_c;
  r[1] = x * y * one_c - z * s;
  r[2] = x * z * one_c + y * s;
  r[3] = y * x * one_c + z * s;
  r[4] = c + y * y * one_c;
  r[5] = y * z * one_c - x * s;
  r[6] = z * x * one_c - y * s;
  r[7] = z * y * one_c + x * s;
  r[8] = c + z * z * one_c;

  out[0] = r[0];
  out[1] = r[3];
  out[2] = r[6];
  out[3] = r[1];
  out[4] = r[4];
  out[5] = r[7];
  out[6] = r[2];
  out[7] = r[5];
  out[8] = r[8];
}

__global__ void erodeDepthKernel(const float* in,
                                 float* out,
                                 int width,
                                 int height,
                                 Config config) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int idx = y * width + x;
  const float center = in[idx];
  if (!validDepth(center, config)) {
    out[idx] = 0.0f;
    return;
  }
  int bad = 0;
  int total = 0;
  const int radius = max(config.erosion_radius, 0);
  for (int yy = max(0, y - radius); yy <= min(height - 1, y + radius); ++yy) {
    for (int xx = max(0, x - radius); xx <= min(width - 1, x + radius); ++xx) {
      const float z = in[yy * width + xx];
      ++total;
      if (!validDepth(z, config) || fabsf(z - center) > config.depth_diff_threshold) {
        ++bad;
      }
    }
  }
  out[idx] = (total > 0 &&
              static_cast<float>(bad) / static_cast<float>(total) >
                  config.erosion_ratio_threshold)
                 ? 0.0f
                 : center;
}

__global__ void bilateralDepthKernel(const float* in,
                                     float* out,
                                     int width,
                                     int height,
                                     Config config) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int radius = max(config.bilateral_radius, 0);
  float mean = 0.0f;
  int valid = 0;
  for (int yy = max(0, y - radius); yy <= min(height - 1, y + radius); ++yy) {
    for (int xx = max(0, x - radius); xx <= min(width - 1, x + radius); ++xx) {
      const float z = in[yy * width + xx];
      if (validDepth(z, config)) {
        mean += z;
        ++valid;
      }
    }
  }
  const int idx = y * width + x;
  if (valid == 0) {
    out[idx] = 0.0f;
    return;
  }
  mean /= static_cast<float>(valid);

  float weighted = 0.0f;
  float weight_sum = 0.0f;
  const float sigma_d2 = 2.0f * config.bilateral_sigma_d * config.bilateral_sigma_d;
  const float sigma_r2 = 2.0f * config.bilateral_sigma_r * config.bilateral_sigma_r;
  for (int yy = max(0, y - radius); yy <= min(height - 1, y + radius); ++yy) {
    for (int xx = max(0, x - radius); xx <= min(width - 1, x + radius); ++xx) {
      const float z = in[yy * width + xx];
      if (!validDepth(z, config)) {
        continue;
      }
      const float dx = static_cast<float>(xx - x);
      const float dy = static_cast<float>(yy - y);
      const float range = z - mean;
      const float w = expf(-(dx * dx + dy * dy) / sigma_d2) *
                      expf(-(range * range) / sigma_r2);
      weighted += w * z;
      weight_sum += w;
    }
  }
  out[idx] = weight_sum > 0.0f ? weighted / weight_sum : 0.0f;
}

__global__ void depthToXyzKernel(const float* depth,
                                 float3* xyz,
                                 int width,
                                 int height,
                                 CameraIntrinsics k,
                                 Config config) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int idx = y * width + x;
  const float z = depth[idx];
  if (!validDepth(z, config)) {
    xyz[idx] = make_float3(0.0f, 0.0f, 0.0f);
    return;
  }
  xyz[idx] = make_float3((static_cast<float>(x) - k.cx) * z / k.fx,
                         (static_cast<float>(y) - k.cy) * z / k.fy, z);
}

__global__ void computeBlockStatsKernel(const float* depth,
                                        const std::uint8_t* mask,
                                        TranslationStats* block_stats,
                                        int width,
                                        int height,
                                        Config config) {
  __shared__ TranslationStats shared[kBlockSize];
  const int tid = threadIdx.x;
  const int global = blockIdx.x * blockDim.x + tid;
  const int pixels = width * height;

  TranslationStats local{width, -1, height, -1, 0, 0.0f};
  if (global < pixels && mask[global] != 0) {
    const int y = global / width;
    const int x = global - y * width;
    local.min_x = x;
    local.max_x = x;
    local.min_y = y;
    local.max_y = y;
    const float z = depth[global];
    if (validDepth(z, config)) {
      local.depth_sum = z;
      local.count = 1;
    }
  }
  shared[tid] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      TranslationStats other = shared[tid + stride];
      shared[tid].min_x = min(shared[tid].min_x, other.min_x);
      shared[tid].max_x = max(shared[tid].max_x, other.max_x);
      shared[tid].min_y = min(shared[tid].min_y, other.min_y);
      shared[tid].max_y = max(shared[tid].max_y, other.max_y);
      shared[tid].count += other.count;
      shared[tid].depth_sum += other.depth_sum;
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_stats[blockIdx.x] = shared[0];
  }
}

__global__ void finalizeTranslationKernel(const TranslationStats* block_stats,
                                          const int* depth_histogram,
                                          int num_blocks,
                                          CameraIntrinsics k,
                                          Config config,
                                          float3* translation) {
  TranslationStats total{INT_MAX, -1, INT_MAX, -1, 0, 0.0f};
  for (int i = 0; i < num_blocks; ++i) {
    TranslationStats s = block_stats[i];
    total.min_x = min(total.min_x, s.min_x);
    total.max_x = max(total.max_x, s.max_x);
    total.min_y = min(total.min_y, s.min_y);
    total.max_y = max(total.max_y, s.max_y);
    total.count += s.count;
    total.depth_sum += s.depth_sum;
  }
  if (total.max_x < total.min_x || total.count <= 0) {
    *translation = make_float3(0.0f, 0.0f, 0.0f);
    return;
  }
  int cumulative = 0;
  int median_bin = 0;
  const int median_rank = total.count / 2;
  for (int bin = 0; bin < kDepthHistogramBins; ++bin) {
    cumulative += depth_histogram[bin];
    if (cumulative > median_rank) {
      median_bin = bin;
      break;
    }
  }
  const float z_range = config.depth_max - config.depth_min;
  const float z = config.depth_min +
                  (static_cast<float>(median_bin) + 0.5f) * z_range /
                      static_cast<float>(kDepthHistogramBins);
  const float u = 0.5f * (static_cast<float>(total.min_x) +
                          static_cast<float>(total.max_x));
  const float v = 0.5f * (static_cast<float>(total.min_y) +
                          static_cast<float>(total.max_y));
  *translation = make_float3((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z);
}

__global__ void depthHistogramKernel(const float* depth,
                                     const std::uint8_t* mask,
                                     int* histogram,
                                     int width,
                                     int height,
                                     Config config) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = width * height;
  if (idx >= pixels || mask[idx] == 0) {
    return;
  }
  const float z = depth[idx];
  if (!validDepth(z, config)) {
    return;
  }
  const float normalized = (z - config.depth_min) / (config.depth_max - config.depth_min);
  const int bin = min(max(static_cast<int>(normalized * kDepthHistogramBins), 0),
                      kDepthHistogramBins - 1);
  atomicAdd(histogram + bin, 1);
}

__global__ void seedPosesKernel(const float* rotations,
                                const float3* translation,
                                float* poses,
                                int batch_size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batch_size) {
    return;
  }
  const float* r = rotations + i * 9;
  float* pose = poses + i * 16;
  pose[0] = r[0];
  pose[1] = r[1];
  pose[2] = r[2];
  pose[3] = translation->x;
  pose[4] = r[3];
  pose[5] = r[4];
  pose[6] = r[5];
  pose[7] = translation->y;
  pose[8] = r[6];
  pose[9] = r[7];
  pose[10] = r[8];
  pose[11] = translation->z;
  pose[12] = 0.0f;
  pose[13] = 0.0f;
  pose[14] = 0.0f;
  pose[15] = 1.0f;
}

__global__ void cropBoxesKernel(const float* poses,
                                int batch_size,
                                CameraIntrinsics k,
                                float diameter,
                                Config config,
                                float4* boxes) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batch_size) {
    return;
  }
  const float* pose = poses + i * 16;
  const float3 center = make_float3(pose[3], pose[7], pose[11]);
  const float radius = diameter * config.crop_ratio * 0.5f;
  const float3 offsets[5] = {
      make_float3(0.0f, 0.0f, 0.0f), make_float3(radius, 0.0f, 0.0f),
      make_float3(-radius, 0.0f, 0.0f), make_float3(0.0f, radius, 0.0f),
      make_float3(0.0f, -radius, 0.0f)};
  float center_u = 0.0f;
  float center_v = 0.0f;
  float radius_px = 1.0f;
  for (int j = 0; j < 5; ++j) {
    const float3 p =
        make_float3(center.x + offsets[j].x, center.y + offsets[j].y,
                    center.z + offsets[j].z);
    const float inv_z = fabsf(p.z) > 1e-12f ? 1.0f / p.z : 0.0f;
    const float u = k.fx * p.x * inv_z + k.cx;
    const float v = k.fy * p.y * inv_z + k.cy;
    if (j == 0) {
      center_u = u;
      center_v = v;
    } else {
      radius_px = fmaxf(radius_px, fabsf(u - center_u));
      radius_px = fmaxf(radius_px, fabsf(v - center_v));
    }
  }
  boxes[i] = make_float4(roundf(center_u - radius_px),
                         roundf(center_v - radius_px),
                         roundf(center_u + radius_px),
                         roundf(center_v + radius_px));
}

__device__ float3 sampleRgbBilinear(const std::uint8_t* rgb,
                                    int width,
                                    int height,
                                    float x,
                                    float y) {
  if (x < 0.0f || y < 0.0f || x > static_cast<float>(width - 1) ||
      y > static_cast<float>(height - 1)) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }
  const int x0 = min(max(static_cast<int>(floorf(x)), 0), width - 1);
  const int y0 = min(max(static_cast<int>(floorf(y)), 0), height - 1);
  const int x1 = min(x0 + 1, width - 1);
  const int y1 = min(y0 + 1, height - 1);
  const float tx = x - static_cast<float>(x0);
  const float ty = y - static_cast<float>(y0);
  const int idx00 = (y0 * width + x0) * 3;
  const int idx10 = (y0 * width + x1) * 3;
  const int idx01 = (y1 * width + x0) * 3;
  const int idx11 = (y1 * width + x1) * 3;
  const float3 c00 = make_float3(static_cast<float>(rgb[idx00 + 0]) / 255.0f,
                                 static_cast<float>(rgb[idx00 + 1]) / 255.0f,
                                 static_cast<float>(rgb[idx00 + 2]) / 255.0f);
  const float3 c10 = make_float3(static_cast<float>(rgb[idx10 + 0]) / 255.0f,
                                 static_cast<float>(rgb[idx10 + 1]) / 255.0f,
                                 static_cast<float>(rgb[idx10 + 2]) / 255.0f);
  const float3 c01 = make_float3(static_cast<float>(rgb[idx01 + 0]) / 255.0f,
                                 static_cast<float>(rgb[idx01 + 1]) / 255.0f,
                                 static_cast<float>(rgb[idx01 + 2]) / 255.0f);
  const float3 c11 = make_float3(static_cast<float>(rgb[idx11 + 0]) / 255.0f,
                                 static_cast<float>(rgb[idx11 + 1]) / 255.0f,
                                 static_cast<float>(rgb[idx11 + 2]) / 255.0f);
  return make_float3((1.0f - ty) * ((1.0f - tx) * c00.x + tx * c10.x) +
                         ty * ((1.0f - tx) * c01.x + tx * c11.x),
                     (1.0f - ty) * ((1.0f - tx) * c00.y + tx * c10.y) +
                         ty * ((1.0f - tx) * c01.y + tx * c11.y),
                     (1.0f - ty) * ((1.0f - tx) * c00.z + tx * c10.z) +
                         ty * ((1.0f - tx) * c01.z + tx * c11.z));
}

__device__ float3 sampleXyzNearest(const float3* xyz,
                                   int width,
                                   int height,
                                   float x,
                                   float y) {
  const int xx = static_cast<int>(roundf(x));
  const int yy = static_cast<int>(roundf(y));
  if (xx < 0 || yy < 0 || xx >= width || yy >= height) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }
  return xyz[yy * width + xx];
}

__device__ void writeNchw(float* dst,
                          int sample,
                          int channel,
                          int y,
                          int x,
                          int channels,
                          int height,
                          int width,
                          float value) {
  dst[((sample * channels + channel) * height + y) * width + x] = value;
}

__global__ void prepareNetworkInputsKernel(const std::uint8_t* rgb,
                                           const float3* xyz_map,
                                           const float3* rendered_rgb,
                                           const float3* rendered_xyz,
                                           const float* poses,
                                           const float4* crop_boxes,
                                           int batch_size,
                                           int image_width,
                                           int image_height,
                                           int crop_width,
                                           int crop_height,
                                           float diameter,
                                           float invalid_depth_threshold,
                                           float* network_rendered,
                                           float* network_observed) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels = batch_size * crop_width * crop_height;
  if (linear >= pixels) {
    return;
  }
  const int x = linear % crop_width;
  const int y = (linear / crop_width) % crop_height;
  const int sample = linear / (crop_width * crop_height);
  const float4 box = crop_boxes[sample];
  const float crop_w = fmaxf(box.z - box.x, 1.0f);
  const float crop_h = fmaxf(box.w - box.y, 1.0f);
  float src_x = box.x + static_cast<float>(x) * crop_w /
                            static_cast<float>(crop_width);
  float src_y = box.y + static_cast<float>(y) * crop_h /
                            static_cast<float>(crop_height);
  if (image_width > 1) {
    src_x = src_x * static_cast<float>(image_width) /
                static_cast<float>(image_width - 1) -
            0.5f;
  }
  if (image_height > 1) {
    src_y = src_y * static_cast<float>(image_height) /
                static_cast<float>(image_height - 1) -
            0.5f;
  }
  const float* pose = poses + sample * 16;
  const float3 t = make_float3(pose[3], pose[7], pose[11]);
  const float radius = fmaxf(diameter * 0.5f, 1e-6f);

  const float3 obs_rgb = sampleRgbBilinear(rgb, image_width, image_height, src_x, src_y);
  float3 obs_xyz = sampleXyzNearest(xyz_map, image_width, image_height, src_x, src_y);
  const bool obs_invalid = obs_xyz.z < invalid_depth_threshold;
  obs_xyz = make_float3((obs_xyz.x - t.x) / radius, (obs_xyz.y - t.y) / radius,
                        (obs_xyz.z - t.z) / radius);
  if (obs_invalid || fabsf(obs_xyz.x) >= 2.0f || fabsf(obs_xyz.y) >= 2.0f ||
      fabsf(obs_xyz.z) >= 2.0f) {
    obs_xyz = make_float3(0.0f, 0.0f, 0.0f);
  }

  const int nhwc = (sample * crop_height + y) * crop_width + x;
  float3 rend_rgb = rendered_rgb[nhwc];
  float3 rend_xyz = rendered_xyz[nhwc];
  const bool rendered_bg = rend_xyz.z < invalid_depth_threshold;
  rend_xyz = make_float3((rend_xyz.x - t.x) / radius, (rend_xyz.y - t.y) / radius,
                         (rend_xyz.z - t.z) / radius);
  if (rendered_bg) {
    rend_xyz = make_float3(0.0f, 0.0f, 0.0f);
  }

  writeNchw(network_rendered, sample, 0, y, x, 6, crop_height, crop_width, rend_rgb.x);
  writeNchw(network_rendered, sample, 1, y, x, 6, crop_height, crop_width, rend_rgb.y);
  writeNchw(network_rendered, sample, 2, y, x, 6, crop_height, crop_width, rend_rgb.z);
  writeNchw(network_rendered, sample, 3, y, x, 6, crop_height, crop_width, rend_xyz.x);
  writeNchw(network_rendered, sample, 4, y, x, 6, crop_height, crop_width, rend_xyz.y);
  writeNchw(network_rendered, sample, 5, y, x, 6, crop_height, crop_width, rend_xyz.z);

  writeNchw(network_observed, sample, 0, y, x, 6, crop_height, crop_width, obs_rgb.x);
  writeNchw(network_observed, sample, 1, y, x, 6, crop_height, crop_width, obs_rgb.y);
  writeNchw(network_observed, sample, 2, y, x, 6, crop_height, crop_width, obs_rgb.z);
  writeNchw(network_observed, sample, 3, y, x, 6, crop_height, crop_width, obs_xyz.x);
  writeNchw(network_observed, sample, 4, y, x, 6, crop_height, crop_width, obs_xyz.y);
  writeNchw(network_observed, sample, 5, y, x, 6, crop_height, crop_width, obs_xyz.z);
}

__global__ void applyPoseDeltasKernel(float* poses,
                                      const float* delta_t,
                                      const float* delta_r,
                                      int batch_size,
                                      float diameter,
                                      Config config) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batch_size) {
    return;
  }
  float* pose = poses + i * 16;
  const float3 omega = make_float3(tanhf(delta_r[i * 3 + 0]) *
                                       config.rotation_normalizer_rad,
                                   tanhf(delta_r[i * 3 + 1]) *
                                       config.rotation_normalizer_rad,
                                   tanhf(delta_r[i * 3 + 2]) *
                                       config.rotation_normalizer_rad);
  float dR[9];
  so3ExpTranspose(omega, dR);
  const float old_r[9] = {pose[0], pose[1], pose[2], pose[4], pose[5],
                          pose[6], pose[8], pose[9], pose[10]};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      float v = 0.0f;
      for (int k = 0; k < 3; ++k) {
        v += dR[row * 3 + k] * old_r[k * 3 + col];
      }
      pose[row * 4 + col] = v;
    }
  }
  const float scale = diameter * 0.5f;
  pose[3] += delta_t[i * 3 + 0] * scale;
  pose[7] += delta_t[i * 3 + 1] * scale;
  pose[11] += delta_t[i * 3 + 2] * scale;
}

__global__ void selectBestPoseKernel(const float* poses,
                                     const float* scores,
                                     int batch_size,
                                     Vec3f center,
                                     DevicePoseResult* result) {
  int best = 0;
  float best_score = scores[0];
  for (int i = 1; i < batch_size; ++i) {
    const float s = scores[i];
    if (s > best_score) {
      best_score = s;
      best = i;
    }
  }
  const float* src = poses + best * 16;
  float* dst_pose = reinterpret_cast<float*>(&result->pose);
  for (int i = 0; i < 16; ++i) {
    dst_pose[i] = 0.0f;
  }
  dst_pose[15] = 1.0f;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      dst_pose[r * 4 + c] = src[r * 4 + c];
    }
  }
  const float neg_center[3] = {-center.x, -center.y, -center.z};
  for (int r = 0; r < 3; ++r) {
    dst_pose[r * 4 + 3] = src[r * 4 + 3] + src[r * 4 + 0] * neg_center[0] +
                           src[r * 4 + 1] * neg_center[1] +
                           src[r * 4 + 2] * neg_center[2];
  }
  result->score = best_score;
  result->index = best;
}

}  // namespace

GpuWorkspace::GpuWorkspace(const Config& config, int max_hyp)
    : max_width(config.max_image_width),
      max_height(config.max_image_height),
      max_pixels(config.max_image_width * config.max_image_height),
      max_hypotheses(max_hyp),
      crop_width(config.input_width),
      crop_height(config.input_height),
      crop_pixels(config.input_width * config.input_height),
      stats_blocks((max_pixels + kBlockSize - 1) / kBlockSize) {
  if (max_width <= 0 || max_height <= 0 || max_hypotheses <= 0 ||
      crop_width <= 0 || crop_height <= 0) {
    throw FoundationPoseError("Invalid GPU workspace dimensions");
  }
  rgb_u8.allocate(static_cast<std::size_t>(max_pixels) * 3);
  mask_u8.allocate(static_cast<std::size_t>(max_pixels));
  depth_raw.allocate(static_cast<std::size_t>(max_pixels) * sizeof(float));
  depth_eroded.allocate(static_cast<std::size_t>(max_pixels) * sizeof(float));
  depth_filtered.allocate(static_cast<std::size_t>(max_pixels) * sizeof(float));
  xyz_map.allocate(static_cast<std::size_t>(max_pixels) * sizeof(float3));
  block_stats.allocate(static_cast<std::size_t>(stats_blocks) * sizeof(TranslationStats));
  depth_histogram.allocate(static_cast<std::size_t>(kDepthHistogramBins) * sizeof(int));
  init_translation.allocate(sizeof(float3));
  poses.allocate(static_cast<std::size_t>(max_hypotheses) * 16 * sizeof(float));
  previous_centered_pose.allocate(16 * sizeof(float));
  crop_boxes.allocate(static_cast<std::size_t>(max_hypotheses) * sizeof(float4));
  rendered_rgb.allocate(static_cast<std::size_t>(max_hypotheses) * crop_pixels *
                        sizeof(float3));
  rendered_xyz.allocate(static_cast<std::size_t>(max_hypotheses) * crop_pixels *
                        sizeof(float3));
  network_rendered.allocate(static_cast<std::size_t>(max_hypotheses) * 6 *
                            crop_pixels * sizeof(float));
  network_observed.allocate(static_cast<std::size_t>(max_hypotheses) * 6 *
                            crop_pixels * sizeof(float));
  delta_translation.allocate(static_cast<std::size_t>(max_hypotheses) * 3 *
                             sizeof(float));
  delta_rotation.allocate(static_cast<std::size_t>(max_hypotheses) * 3 *
                          sizeof(float));
  scores.allocate(static_cast<std::size_t>(max_hypotheses) * sizeof(float));
  best_result.allocate(sizeof(DevicePoseResult));
}

void uploadMeshToDevice(const Mesh& mesh, DeviceMesh& dst, cudaStream_t stream) {
  dst.num_vertices = static_cast<int>(mesh.vertices.size());
  dst.num_faces = static_cast<int>(mesh.faces.size());
  dst.texture_width = mesh.texture_width;
  dst.texture_height = mesh.texture_height;
  dst.has_texture = mesh.hasTexture();
  dst.has_uvs = !mesh.uvs.empty();
  std::vector<float3> vertices;
  std::vector<float3> normals;
  std::vector<float2> uvs;
  std::vector<float3> colors;
  std::vector<int3> faces;
  vertices.reserve(mesh.vertices.size());
  normals.reserve(mesh.vertices.size());
  uvs.reserve(mesh.vertices.size());
  colors.reserve(mesh.vertices.size());
  faces.reserve(mesh.faces.size());

  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    const Vec3f v = mesh.vertices[i];
    vertices.push_back(make_float3(v.x, v.y, v.z));
    const Vec3f n = i < mesh.normals.size() ? mesh.normals[i] : Vec3f{0.0f, 0.0f, 1.0f};
    normals.push_back(make_float3(n.x, n.y, n.z));
    const Vec2f uv = i < mesh.uvs.size() ? mesh.uvs[i] : Vec2f{0.0f, 0.0f};
    uvs.push_back(make_float2(uv.x, uv.y));
    const Vec3u8 c =
        i < mesh.vertex_colors.size() ? mesh.vertex_colors[i] : Vec3u8{128, 128, 128};
    colors.push_back(make_float3(static_cast<float>(c.r) / 255.0f,
                                 static_cast<float>(c.g) / 255.0f,
                                 static_cast<float>(c.b) / 255.0f));
  }
  for (const auto& f : mesh.faces) {
    faces.push_back(make_int3(static_cast<int>(f[0]), static_cast<int>(f[1]),
                              static_cast<int>(f[2])));
  }
  dst.vertices.allocate(vertices.size() * sizeof(float3));
  dst.normals.allocate(normals.size() * sizeof(float3));
  dst.uvs.allocate(uvs.size() * sizeof(float2));
  dst.colors.allocate(colors.size() * sizeof(float3));
  dst.faces.allocate(faces.size() * sizeof(int3));
  checkCuda(cudaMemcpyAsync(dst.vertices.data(), vertices.data(),
                            vertices.size() * sizeof(float3),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync vertices");
  checkCuda(cudaMemcpyAsync(dst.normals.data(), normals.data(),
                            normals.size() * sizeof(float3), cudaMemcpyHostToDevice,
                            stream),
            "cudaMemcpyAsync normals");
  checkCuda(cudaMemcpyAsync(dst.uvs.data(), uvs.data(), uvs.size() * sizeof(float2),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync uvs");
  checkCuda(cudaMemcpyAsync(dst.colors.data(), colors.data(),
                            colors.size() * sizeof(float3), cudaMemcpyHostToDevice,
                            stream),
            "cudaMemcpyAsync colors");
  checkCuda(cudaMemcpyAsync(dst.faces.data(), faces.data(), faces.size() * sizeof(int3),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync faces");
  if (mesh.hasTexture()) {
    std::vector<float3> tex;
    tex.reserve(mesh.texture_rgb.size());
    for (Vec3u8 c : mesh.texture_rgb) {
      tex.push_back(make_float3(static_cast<float>(c.r) / 255.0f,
                                static_cast<float>(c.g) / 255.0f,
                                static_cast<float>(c.b) / 255.0f));
    }
    dst.texture_rgb.allocate(tex.size() * sizeof(float3));
    checkCuda(cudaMemcpyAsync(dst.texture_rgb.data(), tex.data(),
                              tex.size() * sizeof(float3), cudaMemcpyHostToDevice,
                              stream),
              "cudaMemcpyAsync texture");
  }
}

void uploadRotationGridToDevice(const std::vector<Mat3f>& rotations,
                                GpuWorkspace& workspace,
                                cudaStream_t stream) {
  workspace.rotation_count = static_cast<int>(rotations.size());
  std::vector<float> host;
  host.reserve(rotations.size() * 9);
  for (const Mat3f& r : rotations) {
    host.insert(host.end(), r.values.begin(), r.values.end());
  }
  workspace.rotations.allocate(host.size() * sizeof(float));
  checkCuda(cudaMemcpyAsync(workspace.rotations.data(), host.data(),
                            host.size() * sizeof(float), cudaMemcpyHostToDevice,
                            stream),
            "cudaMemcpyAsync rotations");
}

void uploadFrameToDevice(const std::uint8_t* rgb_u8,
                         const float* depth_m,
                         const std::uint8_t* mask_u8,
                         int width,
                         int height,
                         GpuWorkspace& workspace,
                         cudaStream_t stream) {
  // cudaMemcpyDefault resolves the source location through unified virtual
  // addressing, so callers may pass host, device, or managed pointers for the
  // frame buffers. Device pointers must be valid on (or peer-accessible from)
  // the estimator's device.
  const std::size_t pixels = static_cast<std::size_t>(width) * height;
  checkCuda(cudaMemcpyAsync(workspace.rgb_u8.data(), rgb_u8, pixels * 3,
                            cudaMemcpyDefault, stream),
            "cudaMemcpyAsync rgb");
  checkCuda(cudaMemcpyAsync(workspace.depth_raw.data(), depth_m,
                            pixels * sizeof(float), cudaMemcpyDefault,
                            stream),
            "cudaMemcpyAsync depth");
  if (mask_u8 != nullptr) {
    checkCuda(cudaMemcpyAsync(workspace.mask_u8.data(), mask_u8, pixels,
                              cudaMemcpyDefault, stream),
              "cudaMemcpyAsync mask");
  }
}

void runDepthPreprocess(GpuWorkspace& workspace,
                        int width,
                        int height,
                        const Config& config,
                        cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  erodeDepthKernel<<<grid, block, 0, stream>>>(
      workspace.depth_raw.as<float>(), workspace.depth_eroded.as<float>(), width,
      height, config);
  checkCuda(cudaGetLastError(), "erodeDepthKernel");
  bilateralDepthKernel<<<grid, block, 0, stream>>>(
      workspace.depth_eroded.as<float>(), workspace.depth_filtered.as<float>(), width,
      height, config);
  checkCuda(cudaGetLastError(), "bilateralDepthKernel");
}

void runDepthToXyz(GpuWorkspace& workspace,
                   int width,
                   int height,
                   CameraIntrinsics intrinsics,
                   const Config& config,
                   cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  depthToXyzKernel<<<grid, block, 0, stream>>>(
      workspace.depth_filtered.as<float>(), workspace.xyz_map.as<float3>(), width,
      height, intrinsics, config);
  checkCuda(cudaGetLastError(), "depthToXyzKernel");
}

void runInitialTranslation(GpuWorkspace& workspace,
                           int width,
                           int height,
                           CameraIntrinsics intrinsics,
                           const Config& config,
                           cudaStream_t stream) {
  const int pixels = width * height;
  const int blocks = (pixels + kBlockSize - 1) / kBlockSize;
  checkCuda(cudaMemsetAsync(workspace.depth_histogram.data(), 0,
                            static_cast<std::size_t>(kDepthHistogramBins) *
                                sizeof(int),
                            stream),
            "cudaMemsetAsync depth histogram");
  computeBlockStatsKernel<<<blocks, kBlockSize, 0, stream>>>(
      workspace.depth_filtered.as<float>(), workspace.mask_u8.as<std::uint8_t>(),
      workspace.block_stats.as<TranslationStats>(), width, height, config);
  checkCuda(cudaGetLastError(), "computeBlockStatsKernel");
  depthHistogramKernel<<<blocks, kBlockSize, 0, stream>>>(
      workspace.depth_filtered.as<float>(), workspace.mask_u8.as<std::uint8_t>(),
      workspace.depth_histogram.as<int>(), width, height, config);
  checkCuda(cudaGetLastError(), "depthHistogramKernel");
  finalizeTranslationKernel<<<1, 1, 0, stream>>>(
      workspace.block_stats.as<TranslationStats>(), workspace.depth_histogram.as<int>(),
      blocks, intrinsics, config,
      workspace.init_translation.as<float3>());
  checkCuda(cudaGetLastError(), "finalizeTranslationKernel");
}

void seedPosesFromRotations(GpuWorkspace& workspace,
                            int batch_size,
                            cudaStream_t stream) {
  const int blocks = (batch_size + kBlockSize - 1) / kBlockSize;
  seedPosesKernel<<<blocks, kBlockSize, 0, stream>>>(
      workspace.rotations.as<float>(), workspace.init_translation.as<float3>(),
      workspace.poses.as<float>(), batch_size);
  checkCuda(cudaGetLastError(), "seedPosesKernel");
}

void computeCropBoxesOnDevice(const float* poses,
                              int batch_size,
                              CameraIntrinsics intrinsics,
                              float diameter,
                              const Config& config,
                              float* crop_boxes,
                              cudaStream_t stream) {
  const int blocks = (batch_size + kBlockSize - 1) / kBlockSize;
  cropBoxesKernel<<<blocks, kBlockSize, 0, stream>>>(
      poses, batch_size, intrinsics, diameter, config,
      reinterpret_cast<float4*>(crop_boxes));
  checkCuda(cudaGetLastError(), "cropBoxesKernel");
}

void prepareNetworkInputsOnDevice(GpuWorkspace& workspace,
                                  const float* poses,
                                  int batch_size,
                                  int image_width,
                                  int image_height,
                                  float diameter,
                                  float invalid_depth_threshold,
                                  cudaStream_t stream) {
  const int pixels = batch_size * workspace.crop_pixels;
  const int blocks = (pixels + kBlockSize - 1) / kBlockSize;
  prepareNetworkInputsKernel<<<blocks, kBlockSize, 0, stream>>>(
      workspace.rgb_u8.as<std::uint8_t>(), workspace.xyz_map.as<float3>(),
      workspace.rendered_rgb.as<float3>(), workspace.rendered_xyz.as<float3>(),
      poses, workspace.crop_boxes.as<float4>(), batch_size, image_width, image_height,
      workspace.crop_width, workspace.crop_height, diameter, invalid_depth_threshold,
      workspace.network_rendered.as<float>(), workspace.network_observed.as<float>());
  checkCuda(cudaGetLastError(), "prepareNetworkInputsKernel");
}

void applyPoseDeltasOnDevice(float* poses,
                             const float* delta_translation,
                             const float* delta_rotation,
                             int batch_size,
                             float diameter,
                             const Config& config,
                             cudaStream_t stream) {
  const int blocks = (batch_size + kBlockSize - 1) / kBlockSize;
  applyPoseDeltasKernel<<<blocks, kBlockSize, 0, stream>>>(
      poses, delta_translation, delta_rotation, batch_size, diameter, config);
  checkCuda(cudaGetLastError(), "applyPoseDeltasKernel");
}

void selectBestPoseOnDevice(const float* poses,
                            const float* scores,
                            int batch_size,
                            Vec3f mesh_center,
                            DevicePoseResult* result,
                            cudaStream_t stream) {
  selectBestPoseKernel<<<1, 1, 0, stream>>>(poses, scores, batch_size, mesh_center, result);
  checkCuda(cudaGetLastError(), "selectBestPoseKernel");
}

void copyPoseOnDevice(const float* src_pose, float* dst_pose, cudaStream_t stream) {
  checkCuda(cudaMemcpyAsync(dst_pose, src_pose, 16 * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync device pose");
}

}  // namespace foundation_pose_nvidia
