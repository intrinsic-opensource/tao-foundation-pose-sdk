/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared CUDA kernels for the renderer implementations: per-hypothesis vertex
// transform into crop-box clip space, and deferred shading from a rasterized
// barycentric buffer. Included by both the nvdiffrast-backed renderer and the
// built-in rasterizer, which produce an identical "rast" buffer:
//
//   float4 per pixel, rows stored bottom-up (row 0 = NDC y ~= -1):
//     .x = barycentric weight of face vertex 0 (saturated, diagonal-clamped)
//     .y = barycentric weight of face vertex 1
//     .z = interpolated clip z/w, clamped to [-1, 1]
//     .w = float(triangle_index + 1); 0 (all-zero pixel) = background
//
// All kernels are `static` so each including translation unit gets internal
// linkage (the two renderer TUs may coexist in one library).

#pragma once

#include <cuda_runtime_api.h>

#include "foundation_pose_nvidia/config.hpp"
#include "foundation_pose_nvidia/types.hpp"

namespace foundation_pose_nvidia {
namespace raster_kernels {

// Triangle-index codec for the rast buffer's .w channel. Indices are stored as
// plain float values, which round-trip exactly for idx <= 2^24 (16,777,216).
// reserveForMesh guards the mesh size, so the exact-integer range is never
// exceeded.
__device__ __forceinline__ float triIdxToFloat(int idx) {
  return static_cast<float>(idx);
}
__device__ __forceinline__ int floatToTriIdx(float value) {
  return static_cast<int>(value);
}

__device__ __forceinline__ float3 normalize3(float3 v) {
  const float n = sqrtf(fmaxf(v.x * v.x + v.y * v.y + v.z * v.z, 0.0f));
  if (n < 1e-12f) {
    return make_float3(0.0f, 0.0f, 1.0f);
  }
  return make_float3(v.x / n, v.y / n, v.z / n);
}

// Transforms every mesh vertex by every hypothesis pose: object -> camera ->
// crop-box clip space. Outputs per-(sample, vertex) clip positions (float4,
// w = camera z clamped to znear, always > 0 -- so no near-plane clipping is
// ever required downstream) plus camera-space positions/normals for shading.
static __global__ void buildRenderVerticesKernel(const float3* vertices,
                                                 const float3* normals,
                                                 const float* poses,
                                                 const float4* crop_boxes,
                                                 int num_vertices,
                                                 int batch_size,
                                                 CameraIntrinsics k,
                                                 Config config,
                                                 float4* clip_positions,
                                                 float3* camera_positions,
                                                 float3* camera_normals) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * num_vertices;
  if (linear >= total) {
    return;
  }
  const int vertex_idx = linear % num_vertices;
  const int sample = linear / num_vertices;
  const float* pose = poses + sample * 16;
  const float3 p = vertices[vertex_idx];
  const float3 n = normals[vertex_idx];

  const float3 cam = make_float3(pose[0] * p.x + pose[1] * p.y + pose[2] * p.z + pose[3],
                                 pose[4] * p.x + pose[5] * p.y + pose[6] * p.z + pose[7],
                                 pose[8] * p.x + pose[9] * p.y + pose[10] * p.z + pose[11]);
  const float3 normal_cam =
      normalize3(make_float3(pose[0] * n.x + pose[1] * n.y + pose[2] * n.z,
                             pose[4] * n.x + pose[5] * n.y + pose[6] * n.z,
                             pose[8] * n.x + pose[9] * n.y + pose[10] * n.z));
  camera_positions[linear] = cam;
  camera_normals[linear] = normal_cam;

  const float z = fmaxf(cam.z, config.znear);
  const float u = k.fx * cam.x / z + k.cx;
  const float v = k.fy * cam.y / z + k.cy;
  const float4 box = crop_boxes[sample];
  const float crop_w = fmaxf(box.z - box.x, 1.0f);
  const float crop_h = fmaxf(box.w - box.y, 1.0f);
  const float ndc_x = 2.0f * (u - box.x) / crop_w - 1.0f;
  const float ndc_y = 1.0f - 2.0f * (v - box.y) / crop_h;
  const float depth = config.zfar - config.znear;
  const float clip_z = ((config.zfar + config.znear) / depth) * z -
                       (2.0f * config.zfar * config.znear / depth);
  clip_positions[linear] = make_float4(ndc_x * z, ndc_y * z, clip_z, z);
}

__device__ __forceinline__ float3 sampleTexture(const float3* texture,
                                                int width,
                                                int height,
                                                float2 uv) {
  const float u = uv.x - floorf(uv.x);
  const float v = 1.0f - (uv.y - floorf(uv.y));
  const float x = u * static_cast<float>(width) - 0.5f;
  const float y = v * static_cast<float>(height) - 0.5f;
  const int ix0 = static_cast<int>(floorf(x));
  const int iy0 = static_cast<int>(floorf(y));
  const int x0 = (ix0 % width + width) % width;
  const int y0 = (iy0 % height + height) % height;
  const int x1 = (x0 + 1) % width;
  const int y1 = (y0 + 1) % height;
  const float tx = x - static_cast<float>(ix0);
  const float ty = y - static_cast<float>(iy0);
  const float3 c00 = texture[y0 * width + x0];
  const float3 c10 = texture[y0 * width + x1];
  const float3 c01 = texture[y1 * width + x0];
  const float3 c11 = texture[y1 * width + x1];
  return make_float3((1.0f - ty) * ((1.0f - tx) * c00.x + tx * c10.x) +
                         ty * ((1.0f - tx) * c01.x + tx * c11.x),
                     (1.0f - ty) * ((1.0f - tx) * c00.y + tx * c10.y) +
                         ty * ((1.0f - tx) * c01.y + tx * c11.y),
                     (1.0f - ty) * ((1.0f - tx) * c00.z + tx * c10.z) +
                         ty * ((1.0f - tx) * c01.z + tx * c11.z));
}

// Deferred shading: reads the rast buffer (see header comment for layout),
// interpolates camera-space position/normal/albedo with perspective-correct
// barycentrics, applies the ambient+diffuse model, and writes the top-down
// NHWC float3 RGB and XYZ crops the network-input packing consumes.
static __global__ void shadeRasterKernel(const float4* rast,
                                         const int3* faces,
                                         const float3* camera_positions,
                                         const float3* camera_normals,
                                         const float2* uvs,
                                         const float3* vertex_colors,
                                         const float3* texture,
                                         int texture_width,
                                         int texture_height,
                                         int num_vertices,
                                         int num_faces,
                                         int batch_size,
                                         int width,
                                         int height,
                                         bool has_texture,
                                         bool has_uvs,
                                         Config config,
                                         float3* out_rgb,
                                         float3* out_xyz) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * width * height;
  if (linear >= total) {
    return;
  }
  const int x = linear % width;
  const int y = (linear / width) % height;
  const int sample = linear / (width * height);
  const int raster_y = height - 1 - y;
  const float4 r = rast[(sample * height + raster_y) * width + x];
  const int tri_idx = floatToTriIdx(r.w) - 1;
  if (tri_idx < 0 || tri_idx >= num_faces) {
    out_rgb[linear] = make_float3(0.0f, 0.0f, 0.0f);
    out_xyz[linear] = make_float3(0.0f, 0.0f, 0.0f);
    return;
  }
  const int3 face = faces[tri_idx];
  const int v0 = sample * num_vertices + face.x;
  const int v1 = sample * num_vertices + face.y;
  const int v2 = sample * num_vertices + face.z;
  const float b0 = r.x;
  const float b1 = r.y;
  const float b2 = fmaxf(0.0f, 1.0f - b0 - b1);
  const float3 p0 = camera_positions[v0];
  const float3 p1 = camera_positions[v1];
  const float3 p2 = camera_positions[v2];
  const float3 xyz = make_float3(b0 * p0.x + b1 * p1.x + b2 * p2.x,
                                 b0 * p0.y + b1 * p1.y + b2 * p2.y,
                                 b0 * p0.z + b1 * p1.z + b2 * p2.z);
  const float3 n0 = camera_normals[v0];
  const float3 n1 = camera_normals[v1];
  const float3 n2 = camera_normals[v2];
  const float3 n = normalize3(make_float3(b0 * n0.x + b1 * n1.x + b2 * n2.x,
                                          b0 * n0.y + b1 * n1.y + b2 * n2.y,
                                          b0 * n0.z + b1 * n1.z + b2 * n2.z));

  float3 albedo;
  if (has_texture && has_uvs && texture != nullptr && texture_width > 0 &&
      texture_height > 0) {
    const float2 uv0 = uvs[face.x];
    const float2 uv1 = uvs[face.y];
    const float2 uv2 = uvs[face.z];
    albedo = sampleTexture(texture, texture_width, texture_height,
                           make_float2(b0 * uv0.x + b1 * uv1.x + b2 * uv2.x,
                                       b0 * uv0.y + b1 * uv1.y + b2 * uv2.y));
  } else {
    const float3 c0 = vertex_colors[face.x];
    const float3 c1 = vertex_colors[face.y];
    const float3 c2 = vertex_colors[face.z];
    albedo = make_float3(b0 * c0.x + b1 * c1.x + b2 * c2.x,
                         b0 * c0.y + b1 * c1.y + b2 * c2.y,
                         b0 * c0.z + b1 * c1.z + b2 * c2.z);
  }
  const float diffuse = fmaxf(0.0f, -n.z);
  const float light = fminf(fmaxf(config.ambient_weight +
                                      config.diffuse_weight * diffuse,
                                  0.0f),
                            1.5f);
  out_rgb[linear] = make_float3(fminf(fmaxf(albedo.x * light, 0.0f), 1.0f),
                                fminf(fmaxf(albedo.y * light, 0.0f), 1.0f),
                                fminf(fmaxf(albedo.z * light, 0.0f), 1.0f));
  out_xyz[linear] = xyz;
}

}  // namespace raster_kernels
}  // namespace foundation_pose_nvidia
