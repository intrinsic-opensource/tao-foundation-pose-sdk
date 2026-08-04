/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "foundation_pose_nvidia/nvdiffrast_renderer.hpp"
#include "foundation_pose_nvidia/exception.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include "common/common.h"
#include "common/cudaraster/CudaRaster.hpp"
#include "common/cudaraster/impl/Constants.hpp"
#include "common/rasterize.h"

#include "raster_kernels.cuh"

void RasterizeCudaFwdShaderKernel(const RasterizeCudaFwdShaderParams p);

namespace foundation_pose_nvidia {
namespace {


}  // namespace

struct NvdiffrastRenderer::Impl {
  Impl(const Config& cfg, int device, int max_b)
      : config(cfg),
        device_id(device),
        max_batch(max_b),
        raster(std::make_unique<CR::CudaRaster>()) {}

  Config config;
  int device_id = 0;
  int max_batch = 0;
  int reserved_vertices = 0;
  int reserved_faces = 0;
  std::unique_ptr<CR::CudaRaster> raster;
  DeviceBuffer clip_positions;
  DeviceBuffer camera_positions;
  DeviceBuffer camera_normals;
  DeviceBuffer rast;
  DeviceBuffer rast_db;
};

NvdiffrastRenderer::NvdiffrastRenderer(const Config& config,
                                       int device_id,
                                       int max_batch)
    : impl_(std::make_unique<Impl>(config, device_id, max_batch)) {}

NvdiffrastRenderer::~NvdiffrastRenderer() = default;

void NvdiffrastRenderer::reserveForMesh(int num_vertices, int num_faces) {
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
  impl_->rast.allocate(pixels * sizeof(float4));
  impl_->rast_db.allocate(pixels * sizeof(float4));
}

void NvdiffrastRenderer::render(const DeviceMesh& mesh,
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
    throw FoundationPoseError("nvdiffrast renderer received an empty mesh");
  }
  if (batch_size <= 0 || batch_size > impl_->max_batch) {
    throw FoundationPoseError("nvdiffrast renderer batch exceeds configured maximum");
  }
  if (impl_->reserved_vertices < mesh.num_vertices ||
      impl_->reserved_faces < mesh.num_faces) {
    reserveForMesh(mesh.num_vertices, mesh.num_faces);
  }

  const int width_out = impl_->config.input_width;
  const int height_out = impl_->config.input_height;
  const int total_vertices = batch_size * mesh.num_vertices;
  const int block = 256;
  raster_kernels::buildRenderVerticesKernel<<<(total_vertices + block - 1) / block, block, 0, stream>>>(
      mesh.vertices.as<float3>(), mesh.normals.as<float3>(), poses,
      reinterpret_cast<const float4*>(crop_boxes), mesh.num_vertices, batch_size,
      intrinsics, impl_->config, impl_->clip_positions.as<float4>(),
      impl_->camera_positions.as<float3>(), impl_->camera_normals.as<float3>());
  checkCuda(cudaGetLastError(), "buildRenderVerticesKernel");

  const int height = (height_out + CR_TILE_SIZE - 1) & (-CR_TILE_SIZE);
  const int width = (width_out + CR_TILE_SIZE - 1) & (-CR_TILE_SIZE);
  impl_->raster->setVertexBuffer(impl_->clip_positions.data(), mesh.num_vertices);
  impl_->raster->setIndexBuffer(const_cast<void*>(mesh.faces.data()), mesh.num_faces);
  impl_->raster->setBufferSize(width_out, height_out, batch_size);
  impl_->raster->setRenderModeFlags(0);

  const int tile_count_x = (width + CR_MAXVIEWPORT_SIZE - 1) / CR_MAXVIEWPORT_SIZE;
  const int tile_count_y = (height + CR_MAXVIEWPORT_SIZE - 1) / CR_MAXVIEWPORT_SIZE;
  const int tile_size_x =
      ((width + tile_count_x - 1) / tile_count_x + CR_TILE_SIZE - 1) &
      (-CR_TILE_SIZE);
  const int tile_size_y =
      ((height + tile_count_y - 1) / tile_count_y + CR_TILE_SIZE - 1) &
      (-CR_TILE_SIZE);
  for (int tile_y = 0; tile_y < tile_count_y; ++tile_y) {
    for (int tile_x = 0; tile_x < tile_count_x; ++tile_x) {
      const int offset_x = tile_x * tile_size_x;
      const int offset_y = tile_y * tile_size_y;
      const int size_x =
          (width_out - offset_x) < tile_size_x ? (width_out - offset_x) : tile_size_x;
      const int size_y = (height_out - offset_y) < tile_size_y
                             ? (height_out - offset_y)
                             : tile_size_y;
      impl_->raster->setViewport(size_x, size_y, offset_x, offset_y);
      impl_->raster->deferredClear(0u);
      if (!impl_->raster->drawTriangles(nullptr, false, stream)) {
        throw FoundationPoseError("nvdiffrast drawTriangles failed: subtriangle overflow");
      }
    }
  }

  RasterizeCudaFwdShaderParams params{};
  params.pos = impl_->clip_positions.as<float>();
  params.tri = mesh.faces.as<int>();
  params.in_idx = static_cast<const int*>(impl_->raster->getColorBuffer());
  params.out = impl_->rast.as<float>();
  params.out_db = impl_->rast_db.as<float>();
  params.numTriangles = mesh.num_faces;
  params.numVertices = mesh.num_vertices;
  params.width_in = width;
  params.height_in = height;
  params.width_out = width_out;
  params.height_out = height_out;
  params.depth = batch_size;
  params.instance_mode = 1;
  params.xs = 2.0f / static_cast<float>(width_out);
  params.xo = 1.0f / static_cast<float>(width_out) - 1.0f;
  params.ys = 2.0f / static_cast<float>(height_out);
  params.yo = 1.0f / static_cast<float>(height_out) - 1.0f;
  dim3 shader_block = getLaunchBlockSize(RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_WIDTH,
                                         RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_HEIGHT,
                                         params.width_out, params.height_out);
  dim3 shader_grid =
      getLaunchGridSize(shader_block, params.width_out, params.height_out, params.depth);
  void* kernel_args[] = {&params};
  checkCuda(cudaLaunchKernel(reinterpret_cast<const void*>(RasterizeCudaFwdShaderKernel),
                             shader_grid, shader_block, kernel_args, 0, stream),
            "RasterizeCudaFwdShaderKernel");

  const int pixels = batch_size * width_out * height_out;
  raster_kernels::shadeRasterKernel<<<(pixels + block - 1) / block, block, 0, stream>>>(
      impl_->rast.as<float4>(), mesh.faces.as<int3>(),
      impl_->camera_positions.as<float3>(), impl_->camera_normals.as<float3>(),
      mesh.uvs.as<float2>(), mesh.colors.as<float3>(), mesh.texture_rgb.as<float3>(),
      mesh.texture_width, mesh.texture_height, mesh.num_vertices, mesh.num_faces,
      batch_size, width_out, height_out, mesh.has_texture, mesh.has_uvs, impl_->config,
      out_rgb, out_xyz);
  checkCuda(cudaGetLastError(), "shadeRasterKernel");
}

}  // namespace foundation_pose_nvidia

// C entry point used by the main library to instantiate this renderer via
// dlopen/dlsym. Built into libfoundation_pose_nvdiffrast.so together with the
// nvdiffrast CUDA-rasterizer core (NVIDIA Source Code License); both sides of
// the boundary are built by the same toolchain in this project.
extern "C" foundation_pose_nvidia::IRenderer* fp_plugin_create_renderer(
    const foundation_pose_nvidia::Config* config, int device_id, int max_batch) {
  try {
    return new foundation_pose_nvidia::NvdiffrastRenderer(*config, device_id, max_batch);
  } catch (...) {
    return nullptr;
  }
}
