// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "frame_data.hpp"

#include <png.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "foundation_pose_nvidia/exception.hpp"

namespace sample {

using foundation_pose_nvidia::FoundationPoseError;

namespace {

// 8-bit interleaved read via libpng's simplified API (RGB or GRAY). Returns raw
// sample values (no gamma changes) for 8-bit images.
std::vector<uint8_t> loadPng8(const std::string& path, int& width, int& height,
                              png_uint_32 format) {
  png_image image;
  std::memset(&image, 0, sizeof(image));
  image.version = PNG_IMAGE_VERSION;
  if (png_image_begin_read_from_file(&image, path.c_str()) == 0) {
    throw FoundationPoseError("failed to open PNG: " + path);
  }
  image.format = format;
  std::vector<uint8_t> buffer(PNG_IMAGE_SIZE(image));
  if (png_image_finish_read(&image, nullptr, buffer.data(), 0, nullptr) == 0) {
    png_image_free(&image);
    throw FoundationPoseError("failed to decode PNG: " + path);
  }
  width = static_cast<int>(image.width);
  height = static_cast<int>(image.height);
  return buffer;
}

// 16-bit grayscale read via the low-level API. The simplified API only offers a
// gamma-converted ("linear") 16-bit path, which would corrupt raw depth values,
// so depth must use the low-level reader to get untouched samples.
std::vector<uint16_t> loadPng16Gray(const std::string& path, int& width, int& height) {
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (fp == nullptr) throw FoundationPoseError("failed to open PNG: " + path);

  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png ? png_create_info_struct(png) : nullptr;
  if (png == nullptr || info == nullptr) {
    if (png) png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);
    throw FoundationPoseError("libpng init failed for: " + path);
  }
  std::vector<uint16_t> out;
  std::vector<png_bytep> rows;
  if (setjmp(png_jmpbuf(png))) {  // libpng error handler jumps here
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);
    throw FoundationPoseError("failed to decode 16-bit PNG: " + path);
  }
  png_init_io(png, fp);
  png_read_info(png, info);
  width = static_cast<int>(png_get_image_width(png, info));
  height = static_cast<int>(png_get_image_height(png, info));
  const int bit_depth = png_get_bit_depth(png, info);
  if (png_get_color_type(png, info) != PNG_COLOR_TYPE_GRAY) {
    throw FoundationPoseError("expected grayscale depth PNG: " + path);
  }
  if (bit_depth == 16) png_set_swap(png);  // PNG is big-endian; swap to host order
  png_read_update_info(png, info);

  out.resize(static_cast<size_t>(width) * height);
  rows.resize(height);
  for (int y = 0; y < height; ++y) {
    rows[y] = reinterpret_cast<png_bytep>(&out[static_cast<size_t>(y) * width]);
  }
  png_read_image(png, rows.data());
  png_read_end(png, nullptr);
  png_destroy_read_struct(&png, &info, nullptr);
  std::fclose(fp);
  return out;
}

}  // namespace

std::vector<uint8_t> loadRgb(const std::string& path, int& width, int& height) {
  return loadPng8(path, width, height, PNG_FORMAT_RGB);
}

std::vector<uint8_t> loadMask(const std::string& path, int& width, int& height) {
  return loadPng8(path, width, height, PNG_FORMAT_GRAY);
}

std::vector<float> loadDepthMeters(const std::string& path, float depth_scale,
                                   int& width, int& height) {
  const std::vector<uint16_t> raw = loadPng16Gray(path, width, height);
  std::vector<float> depth_m(raw.size());
  const float to_m = depth_scale / 1000.0f;
  for (size_t i = 0; i < raw.size(); ++i) {
    depth_m[i] = static_cast<float>(raw[i]) * to_m;
  }
  return depth_m;
}

}  // namespace sample
