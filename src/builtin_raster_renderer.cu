/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "foundation_pose_nvidia/builtin_raster_renderer.hpp"
#include "foundation_pose_nvidia/exception.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "raster_kernels.cuh"

namespace foundation_pose_nvidia {
namespace {

constexpr int kBlockSize = 256;
// Triangle indices are stored as exact float values in the rast buffer; the
// packing below also reserves the low 32 bits of the z-buffer key.
constexpr int kMaxTriangles = 1 << 24;
constexpr unsigned long long kEmptyKey = 0xFFFFFFFFFFFFFFFFull;

// Orderable depth key: clip z/w in [-1, 1] mapped to a non-negative float in
// [0, 1], whose IEEE-754 bit pattern is monotonically increasing. Packed with
// (triangle_index + 1) in the low bits so atomicMin resolves equal depths
// deterministically toward the lower triangle index.
__device__ __forceinline__ unsigned long long packDepthKey(float zw, int tri_plus_one) {
  const float d01 = fminf(fmaxf(zw * 0.5f + 0.5f, 0.0f), 1.0f);
  const unsigned int depth_bits = __float_as_uint(d01);
  return (static_cast<unsigned long long>(depth_bits) << 32) |
         static_cast<unsigned int>(tri_plus_one);
}

// One thread per (sample, triangle): rasterize the triangle into the sample's
// z-buffer tile with a pixel-center coverage test in NDC. Rows are bottom-up
// (row py has NDC y = (2*py + 1)/H - 1), matching the rast-buffer contract.
// buildRenderVerticesKernel guarantees w = camera-z >= znear > 0 for every
// vertex, so no near-plane clipping is required.
__global__ void rasterTrianglesKernel(const float4* clip_positions,
                                      const int3* faces,
                                      int num_vertices,
                                      int num_faces,
                                      int batch_size,
                                      int width,
                                      int height,
                                      unsigned long long* zbuffer) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * num_faces;
  if (linear >= total) {
    return;
  }
  const int tri = linear % num_faces;
  const int sample = linear / num_faces;
  const int3 face = faces[tri];
  const int base = sample * num_vertices;
  const float4 p0 = clip_positions[base + face.x];
  const float4 p1 = clip_positions[base + face.y];
  const float4 p2 = clip_positions[base + face.z];

  // Projected pixel coordinates (pixel px center sits at NDC (2*px+1)/W - 1).
  const float half_w = 0.5f * static_cast<float>(width);
  const float half_h = 0.5f * static_cast<float>(height);
  const float x0 = (p0.x / p0.w + 1.0f) * half_w - 0.5f;
  const float x1 = (p1.x / p1.w + 1.0f) * half_w - 0.5f;
  const float x2 = (p2.x / p2.w + 1.0f) * half_w - 0.5f;
  const float y0 = (p0.y / p0.w + 1.0f) * half_h - 0.5f;
  const float y1 = (p1.y / p1.w + 1.0f) * half_h - 0.5f;
  const float y2 = (p2.y / p2.w + 1.0f) * half_h - 0.5f;

  const int min_x = max(0, static_cast<int>(floorf(fminf(fminf(x0, x1), x2))));
  const int max_x = min(width - 1, static_cast<int>(ceilf(fmaxf(fmaxf(x0, x1), x2))));
  const int min_y = max(0, static_cast<int>(floorf(fminf(fminf(y0, y1), y2))));
  const int max_y = min(height - 1, static_cast<int>(ceilf(fmaxf(fmaxf(y0, y1), y2))));
  if (min_x > max_x || min_y > max_y) {
    return;
  }

  unsigned long long* tile =
      zbuffer + static_cast<std::size_t>(sample) * width * height;
  const int tri_plus_one = tri + 1;
  const float inv_w = 1.0f / static_cast<float>(width);
  const float inv_h = 1.0f / static_cast<float>(height);

  for (int py = min_y; py <= max_y; ++py) {
    const float fy = (2.0f * py + 1.0f) * inv_h - 1.0f;
    const float p0y = p0.y - fy * p0.w;
    const float p1y = p1.y - fy * p1.w;
    const float p2y = p2.y - fy * p2.w;
    for (int px = min_x; px <= max_x; ++px) {
      const float fx = (2.0f * px + 1.0f) * inv_w - 1.0f;
      // Homogeneous 2D edge functions (perspective-correct sub-areas).
      const float p0x = p0.x - fx * p0.w;
      const float p1x = p1.x - fx * p1.w;
      const float p2x = p2.x - fx * p2.w;
      const float a0 = p1x * p2y - p1y * p2x;
      const float a1 = p2x * p0y - p2y * p0x;
      const float a2 = p0x * p1y - p0y * p1x;
      const bool inside_pos = a0 >= 0.0f && a1 >= 0.0f && a2 >= 0.0f;
      const bool inside_neg = a0 <= 0.0f && a1 <= 0.0f && a2 <= 0.0f;
      const float area = a0 + a1 + a2;
      if ((!inside_pos && !inside_neg) || area == 0.0f) {
        continue;
      }
      const float z = p0.z * a0 + p1.z * a1 + p2.z * a2;
      const float w = p0.w * a0 + p1.w * a1 + p2.w * a2;
      if (w == 0.0f) {
        continue;
      }
      const float zw = z / w;
      atomicMin(&tile[py * width + px], packDepthKey(zw, tri_plus_one));
    }
  }
}

// One thread per pixel: decode the z-buffer winner and emit the rast float4
// (b0, b1, zw, tri_index + 1) with the same barycentric normalization and
// clamping the shading kernel was validated against; background stays zero.
__global__ void resolveRastKernel(const unsigned long long* zbuffer,
                                  const float4* clip_positions,
                                  const int3* faces,
                                  int num_vertices,
                                  int batch_size,
                                  int width,
                                  int height,
                                  float4* rast) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * width * height;
  if (linear >= total) {
    return;
  }
  const unsigned long long key = zbuffer[linear];
  if (key == kEmptyKey) {
    rast[linear] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }
  const int tri_plus_one = static_cast<int>(key & 0xFFFFFFFFull);
  const int px = linear % width;
  const int py = (linear / width) % height;
  const int sample = linear / (width * height);
  const int3 face = faces[tri_plus_one - 1];
  const int base = sample * num_vertices;
  const float4 p0 = clip_positions[base + face.x];
  const float4 p1 = clip_positions[base + face.y];
  const float4 p2 = clip_positions[base + face.z];

  const float fx = (2.0f * px + 1.0f) / static_cast<float>(width) - 1.0f;
  const float fy = (2.0f * py + 1.0f) / static_cast<float>(height) - 1.0f;
  const float p0x = p0.x - fx * p0.w;
  const float p0y = p0.y - fy * p0.w;
  const float p1x = p1.x - fx * p1.w;
  const float p1y = p1.y - fy * p1.w;
  const float p2x = p2.x - fx * p2.w;
  const float p2y = p2.y - fy * p2.w;
  const float a0 = p1x * p2y - p1y * p2x;
  const float a1 = p2x * p0y - p2y * p0x;
  const float a2 = p0x * p1y - p0y * p1x;

  const float iw = 1.0f / (a0 + a1 + a2);
  float b0 = __saturatef(a0 * iw);
  float b1 = __saturatef(a1 * iw);
  const float bs = 1.0f / fmaxf(b0 + b1, 1.0f);
  b0 *= bs;
  b1 *= bs;

  const float z = p0.z * a0 + p1.z * a1 + p2.z * a2;
  const float w = p0.w * a0 + p1.w * a1 + p2.w * a2;
  const float zw = fmaxf(fminf(z / w, 1.0f), -1.0f);

  rast[linear] = make_float4(b0, b1, zw,
                             raster_kernels::triIdxToFloat(tri_plus_one));
}

}  // namespace

struct BuiltinRasterRenderer::Impl {
  Impl(const Config& cfg, int device, int max_b)
      : config(cfg), device_id(device), max_batch(max_b) {}

  Config config;
  int device_id = 0;
  int max_batch = 0;
  int reserved_vertices = 0;
  int reserved_faces = 0;
  DeviceBuffer clip_positions;
  DeviceBuffer camera_positions;
  DeviceBuffer camera_normals;
  DeviceBuffer zbuffer;
  DeviceBuffer rast;
};

BuiltinRasterRenderer::BuiltinRasterRenderer(const Config& config,
                                             int device_id,
                                             int max_batch)
    : impl_(std::make_unique<Impl>(config, device_id, max_batch)) {}

BuiltinRasterRenderer::~BuiltinRasterRenderer() = default;

void BuiltinRasterRenderer::reserveForMesh(int num_vertices, int num_faces) {
  if (num_faces >= kMaxTriangles) {
    throw FoundationPoseError("BuiltinRasterRenderer supports up to 2^24 triangles, got " +
                             std::to_string(num_faces));
  }
  impl_->reserved_vertices = num_vertices;
  impl_->reserved_faces = num_faces;
  impl_->clip_positions.allocate(static_cast<std::size_t>(impl_->max_batch) *
                                 num_vertices * sizeof(float4));
  impl_->camera_positions.allocate(static_cast<std::size_t>(impl_->max_batch) *
                                   num_vertices * sizeof(float3));
  impl_->camera_normals.allocate(static_cast<std::size_t>(impl_->max_batch) *
                                 num_vertices * sizeof(float3));
  const std::size_t pixels = static_cast<std::size_t>(impl_->max_batch) *
                             impl_->config.input_width * impl_->config.input_height;
  impl_->zbuffer.allocate(pixels * sizeof(unsigned long long));
  impl_->rast.allocate(pixels * sizeof(float4));
}

void BuiltinRasterRenderer::render(const DeviceMesh& mesh,
                                   const float* poses,
                                   const float* crop_boxes,
                                   int batch_size,
                                   CameraIntrinsics intrinsics,
                                   int,
                                   int,
                                   float3* out_rgb,
                                   float3* out_xyz,
                                   cudaStream_t stream) {
  if (mesh.num_vertices <= 0 || mesh.num_faces <= 0) {
    throw FoundationPoseError("builtin renderer received an empty mesh");
  }
  if (batch_size <= 0 || batch_size > impl_->max_batch) {
    throw FoundationPoseError("builtin renderer batch exceeds configured maximum");
  }
  if (impl_->reserved_vertices < mesh.num_vertices ||
      impl_->reserved_faces < mesh.num_faces) {
    reserveForMesh(mesh.num_vertices, mesh.num_faces);
  }

  const int width = impl_->config.input_width;
  const int height = impl_->config.input_height;
  const int total_vertices = batch_size * mesh.num_vertices;
  raster_kernels::buildRenderVerticesKernel<<<
      (total_vertices + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
      mesh.vertices.as<float3>(), mesh.normals.as<float3>(), poses,
      reinterpret_cast<const float4*>(crop_boxes), mesh.num_vertices, batch_size,
      intrinsics, impl_->config, impl_->clip_positions.as<float4>(),
      impl_->camera_positions.as<float3>(), impl_->camera_normals.as<float3>());
  checkCuda(cudaGetLastError(), "buildRenderVerticesKernel");

  const std::size_t pixel_count =
      static_cast<std::size_t>(batch_size) * width * height;
  checkCuda(cudaMemsetAsync(impl_->zbuffer.data(), 0xFF,
                            pixel_count * sizeof(unsigned long long), stream),
            "builtin renderer z-buffer clear");

  const int total_tris = batch_size * mesh.num_faces;
  rasterTrianglesKernel<<<(total_tris + kBlockSize - 1) / kBlockSize, kBlockSize, 0,
                          stream>>>(impl_->clip_positions.as<float4>(),
                                    mesh.faces.as<int3>(), mesh.num_vertices,
                                    mesh.num_faces, batch_size, width, height,
                                    impl_->zbuffer.as<unsigned long long>());
  checkCuda(cudaGetLastError(), "rasterTrianglesKernel");

  const int total_pixels = static_cast<int>(pixel_count);
  resolveRastKernel<<<(total_pixels + kBlockSize - 1) / kBlockSize, kBlockSize, 0,
                      stream>>>(impl_->zbuffer.as<unsigned long long>(),
                                impl_->clip_positions.as<float4>(),
                                mesh.faces.as<int3>(), mesh.num_vertices, batch_size,
                                width, height, impl_->rast.as<float4>());
  checkCuda(cudaGetLastError(), "resolveRastKernel");

  raster_kernels::shadeRasterKernel<<<(total_pixels + kBlockSize - 1) / kBlockSize,
                                      kBlockSize, 0, stream>>>(
      impl_->rast.as<float4>(), mesh.faces.as<int3>(),
      impl_->camera_positions.as<float3>(), impl_->camera_normals.as<float3>(),
      mesh.uvs.as<float2>(), mesh.colors.as<float3>(), mesh.texture_rgb.as<float3>(),
      mesh.texture_width, mesh.texture_height, mesh.num_vertices, mesh.num_faces,
      batch_size, width, height, mesh.has_texture, mesh.has_uvs, impl_->config,
      out_rgb, out_xyz);
  checkCuda(cudaGetLastError(), "shadeRasterKernel");
}

}  // namespace foundation_pose_nvidia
