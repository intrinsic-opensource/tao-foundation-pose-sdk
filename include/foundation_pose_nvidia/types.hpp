/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace foundation_pose_nvidia {

struct Vec2f {
  float x = 0.0f;
  float y = 0.0f;
};

struct Vec3f {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;

  // Hidden-friend arithmetic (found via ADL on Vec3f), kept out of the
  // enclosing namespace's overload set.
  friend Vec3f operator+(Vec3f a, Vec3f b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
  }
  friend Vec3f operator-(Vec3f a, Vec3f b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
  }
  friend Vec3f operator*(Vec3f a, float s) {
    return {a.x * s, a.y * s, a.z * s};
  }
  friend Vec3f operator/(Vec3f a, float s) {
    return {a.x / s, a.y / s, a.z / s};
  }
};

struct Vec3u8 {
  std::uint8_t r = 128;
  std::uint8_t g = 128;
  std::uint8_t b = 128;
};

struct Mat3f {
  std::array<float, 9> values{};

  static Mat3f identity() {
    Mat3f out;
    out(0, 0) = 1.0f;
    out(1, 1) = 1.0f;
    out(2, 2) = 1.0f;
    return out;
  }

  float& operator()(std::size_t row, std::size_t col) {
    return values[row * 3 + col];
  }

  float operator()(std::size_t row, std::size_t col) const {
    return values[row * 3 + col];
  }
};

struct Mat4f {
  std::array<float, 16> values{};

  static Mat4f identity() {
    Mat4f out;
    out(0, 0) = 1.0f;
    out(1, 1) = 1.0f;
    out(2, 2) = 1.0f;
    out(3, 3) = 1.0f;
    return out;
  }

  float& operator()(std::size_t row, std::size_t col) {
    return values[row * 4 + col];
  }

  float operator()(std::size_t row, std::size_t col) const {
    return values[row * 4 + col];
  }
};

struct CameraIntrinsics {
  float fx = 0.0f;
  float fy = 0.0f;
  float cx = 0.0f;
  float cy = 0.0f;

  Mat3f matrix() const {
    Mat3f out = Mat3f::identity();
    out(0, 0) = fx;
    out(1, 1) = fy;
    out(0, 2) = cx;
    out(1, 2) = cy;
    return out;
  }

  static CameraIntrinsics fromRowMajor(const float* k) {
    return CameraIntrinsics{k[0], k[4], k[2], k[5]};
  }
};

struct RgbImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> data;  // HWC, RGB, 8-bit.

  bool empty() const { return width <= 0 || height <= 0 || data.empty(); }
};

struct DepthImage {
  int width = 0;
  int height = 0;
  std::vector<float> meters;  // HW, meters.

  bool empty() const { return width <= 0 || height <= 0 || meters.empty(); }
};

struct MaskImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> values;  // HW, non-zero is object.

  bool empty() const { return width <= 0 || height <= 0 || values.empty(); }
};

struct ModelFreeReferenceView {
  int width = 0;
  int height = 0;
  const std::uint8_t* rgb_u8 = nullptr;   // HWC RGB, width * height * 3 bytes.
  const float* depth_m = nullptr;         // HW float32, meters.
  const std::uint8_t* mask_u8 = nullptr;  // HW uint8, non-zero is object.
  CameraIntrinsics intrinsics{};
  Mat4f camera_to_world = Mat4f::identity();
};

struct Mesh {
  std::vector<Vec3f> vertices;
  std::vector<std::array<std::uint32_t, 3>> faces;
  std::vector<Vec3f> normals;
  std::vector<Vec2f> uvs;
  std::vector<Vec3u8> vertex_colors;
  int texture_width = 0;
  int texture_height = 0;
  std::vector<Vec3u8> texture_rgb;
  std::string source_path;

  bool hasTexture() const {
    return texture_width > 0 && texture_height > 0 && !texture_rgb.empty();
  }
};

struct PreprocessedMesh {
  Mesh centered_mesh;
  float diameter = 0.0f;
  Vec3f center{};
  std::vector<Vec3f> downsampled_points;
};

struct CropBox {
  float left = 0.0f;
  float top = 0.0f;
  float right = 0.0f;
  float bottom = 0.0f;
};

struct Tensor {
  std::string name;
  std::vector<std::int64_t> shape;
  std::vector<float> data;

  std::size_t elementCount() const { return data.size(); }
  std::size_t byteSize() const { return data.size() * sizeof(float); }
};

struct NetworkBatch {
  Tensor rendered;
  Tensor observed;
  std::size_t batch_size = 0;
};

struct RefinementDeltas {
  std::vector<Vec3f> translation;
  std::vector<Vec3f> rotation_axis_angle;
};

struct PoseEstimate {
  Mat4f pose = Mat4f::identity();  // Object-to-camera, original mesh frame.
  float score = 0.0f;
};

}  // namespace foundation_pose_nvidia
