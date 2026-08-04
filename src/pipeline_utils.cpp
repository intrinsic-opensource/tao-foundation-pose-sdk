/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>

#include "math_utils.hpp"

#include "foundation_pose_nvidia/exception.hpp"

namespace foundation_pose_nvidia::detail {

namespace {

struct Triangle {
  int a = 0;
  int b = 0;
  int c = 0;
};

std::vector<float> erodeDepth(const DepthImage& depth, const Config& config) {
  const int w = depth.width;
  const int h = depth.height;
  const int radius = std::max(config.erosion_radius, 0);
  const int kernel = 2 * radius + 1;
  const auto denom = static_cast<float>(kernel * kernel);
  std::vector<float> out(depth.meters.size(), 0.0f);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float center = depth.meters[static_cast<std::size_t>(y) * w + x];
      const bool center_invalid =
          center < config.depth_min || center >= config.depth_max;
      int bad = 0;
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          const int xx = x + dx;
          const int yy = y + dy;
          float neighbor = 0.0f;
          if (xx >= 0 && yy >= 0 && xx < w && yy < h) {
            neighbor = depth.meters[static_cast<std::size_t>(yy) * w + xx];
          }
          const bool bad_neighbor =
              neighbor < config.depth_min || neighbor >= config.depth_max ||
              std::abs(neighbor - center) > config.depth_diff_threshold;
          if (bad_neighbor) {
            ++bad;
          }
        }
      }
      if (!center_invalid &&
          static_cast<float>(bad) / denom <= config.erosion_ratio_threshold) {
        out[static_cast<std::size_t>(y) * w + x] = center;
      }
    }
  }

  return out;
}

std::vector<float> bilateralFilter(const std::vector<float>& depth,
                                   int w,
                                   int h,
                                   const Config& config) {
  const int radius = std::max(config.bilateral_radius, 0);
  std::vector<float> out(depth.size(), 0.0f);
  std::vector<float> spatial;
  spatial.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const auto d2 = static_cast<float>(dx * dx + dy * dy);
      spatial.push_back(std::exp(-d2 / (2.0f * config.bilateral_sigma_d *
                                        config.bilateral_sigma_d)));
    }
  }

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float mean = 0.0f;
      int valid_count = 0;
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          const int xx = x + dx;
          const int yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
            continue;
          }
          const float z = depth[static_cast<std::size_t>(yy) * w + xx];
          if (z >= config.depth_min && z < config.depth_max) {
            mean += z;
            ++valid_count;
          }
        }
      }
      if (valid_count == 0) {
        continue;
      }
      mean /= static_cast<float>(valid_count);

      float weighted = 0.0f;
      float weight_sum = 0.0f;
      std::size_t k = 0;
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx, ++k) {
          const int xx = x + dx;
          const int yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
            continue;
          }
          const float z = depth[static_cast<std::size_t>(yy) * w + xx];
          if (z < config.depth_min || z >= config.depth_max) {
            continue;
          }
          const float range = z - mean;
          const float wr = std::exp(-(range * range) /
                                    (2.0f * config.bilateral_sigma_r *
                                     config.bilateral_sigma_r));
          const float weight = spatial[k] * wr;
          weighted += weight * z;
          weight_sum += weight;
        }
      }
      if (weight_sum > 0.0f) {
        out[static_cast<std::size_t>(y) * w + x] = weighted / weight_sum;
      }
    }
  }
  return out;
}

int midpoint(int a,
             int b,
             std::vector<Vec3f>& vertices,
             std::map<std::pair<int, int>, int>& cache) {
  const auto key = std::minmax(a, b);
  if (const auto found = cache.find(key); found != cache.end()) {
    return found->second;
  }
  const Vec3f mid = normalize((vertices[a] + vertices[b]) * 0.5f);
  const auto idx = static_cast<int>(vertices.size());
  vertices.push_back(mid);
  cache.emplace(key, idx);
  return idx;
}

std::vector<Vec3f> sampleIcosphere(int min_vertices) {
  const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
  std::vector<Vec3f> vertices = {
      normalize({-1.0f, t, 0.0f}),  normalize({1.0f, t, 0.0f}),
      normalize({-1.0f, -t, 0.0f}), normalize({1.0f, -t, 0.0f}),
      normalize({0.0f, -1.0f, t}),  normalize({0.0f, 1.0f, t}),
      normalize({0.0f, -1.0f, -t}), normalize({0.0f, 1.0f, -t}),
      normalize({t, 0.0f, -1.0f}),  normalize({t, 0.0f, 1.0f}),
      normalize({-t, 0.0f, -1.0f}), normalize({-t, 0.0f, 1.0f}),
  };
  std::vector<Triangle> faces = {
      {0, 11, 5},  {0, 5, 1},   {0, 1, 7},   {0, 7, 10},
      {0, 10, 11}, {1, 5, 9},   {5, 11, 4},  {11, 10, 2},
      {10, 7, 6},  {7, 1, 8},   {3, 9, 4},   {3, 4, 2},
      {3, 2, 6},   {3, 6, 8},   {3, 8, 9},   {4, 9, 5},
      {2, 4, 11},  {6, 2, 10},  {8, 6, 7},   {9, 8, 1},
  };

  while (static_cast<int>(vertices.size()) < min_vertices) {
    std::map<std::pair<int, int>, int> cache;
    std::vector<Triangle> next;
    next.reserve(faces.size() * 4);
    for (Triangle f : faces) {
      const int ab = midpoint(f.a, f.b, vertices, cache);
      const int bc = midpoint(f.b, f.c, vertices, cache);
      const int ca = midpoint(f.c, f.a, vertices, cache);
      next.push_back({f.a, ab, ca});
      next.push_back({f.b, bc, ab});
      next.push_back({f.c, ca, bc});
      next.push_back({ab, bc, ca});
    }
    faces = std::move(next);
  }
  return vertices;
}

Mat4f cameraInObjectFromView(Vec3f cam_pos) {
  const Vec3f z_axis = normalize(cam_pos * -1.0f);
  Vec3f x_axis = cross({0.0f, 0.0f, 1.0f}, z_axis);
  if (norm(x_axis) < 0.001f) {
    x_axis = {1.0f, 0.0f, 0.0f};
  }
  x_axis = normalize(x_axis);
  const Vec3f y_axis = normalize(cross(z_axis, x_axis));

  Mat4f out = Mat4f::identity();
  out(0, 0) = x_axis.x;
  out(1, 0) = x_axis.y;
  out(2, 0) = x_axis.z;
  out(0, 1) = y_axis.x;
  out(1, 1) = y_axis.y;
  out(2, 1) = y_axis.z;
  out(0, 2) = z_axis.x;
  out(1, 2) = z_axis.y;
  out(2, 2) = z_axis.z;
  setTranslationPart(out, cam_pos);
  return out;
}

std::vector<Mat3f> generateRotationGrid(const Config& config) {
  const std::vector<Vec3f> views = sampleIcosphere(config.min_n_views);
  std::vector<Mat3f> rotations;
  const int n_inplane =
      std::max(1, static_cast<int>(std::round(360.0f / config.inplane_step_deg)));
  const float step = config.inplane_step_deg * kPi / 180.0f;
  for (Vec3f view : views) {
    const Mat4f cam_in_obj = cameraInObjectFromView(view);
    const Mat3f base_r = rotationPart(cam_in_obj);
    for (int j = 0; j < n_inplane; ++j) {
      const Mat3f cam_r = multiply(base_r, rotationZ(static_cast<float>(j) * step));
      const Mat4f obj_in_cam = inverseRigid(makeTransform(cam_r, translationPart(cam_in_obj)));
      rotations.push_back(rotationPart(obj_in_cam));
    }
  }

  std::vector<Mat3f> clustered;
  const float threshold = config.cluster_threshold_deg * kPi / 180.0f;
  for (const Mat3f& candidate : rotations) {
    bool is_new = true;
    for (const Mat3f& existing : clustered) {
      if (geodesicDistance(candidate, existing) < threshold) {
        is_new = false;
        break;
      }
    }
    if (is_new) {
      clustered.push_back(candidate);
    }
  }
  return clustered;
}

}  // namespace

std::vector<float> preprocessDepth(const DepthImage& depth, const Config& config) {
  if (depth.width <= 0 || depth.height <= 0 ||
      depth.meters.size() != static_cast<std::size_t>(depth.width * depth.height)) {
    throw FoundationPoseError("Invalid depth image shape");
  }
  DepthImage clamped = depth;
  for (float& z : clamped.meters) {
    if (z < config.depth_min || z >= config.depth_max || !std::isfinite(z)) {
      z = 0.0f;
    }
  }
  const std::vector<float> eroded = erodeDepth(clamped, config);
  return bilateralFilter(eroded, depth.width, depth.height, config);
}

std::vector<Vec3f> depthToXyz(const std::vector<float>& depth_m,
                              int width,
                              int height,
                              CameraIntrinsics intrinsics,
                              const Config& config) {
  std::vector<Vec3f> xyz(depth_m.size(), Vec3f{});
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t idx = static_cast<std::size_t>(y) * width + x;
      const float z = depth_m[idx];
      if (z < config.depth_min || z >= config.depth_max) {
        continue;
      }
      xyz[idx] = {
          (static_cast<float>(x) - intrinsics.cx) * z / intrinsics.fx,
          (static_cast<float>(y) - intrinsics.cy) * z / intrinsics.fy,
          z,
      };
    }
  }
  return xyz;
}

Vec3f guessTranslation(const std::vector<float>& depth_m,
                       const MaskImage& mask,
                       CameraIntrinsics intrinsics,
                       const Config& config) {
  if (mask.width <= 0 || mask.height <= 0 ||
      mask.values.size() != static_cast<std::size_t>(mask.width * mask.height)) {
    return {0.0f, 0.0f, 0.0f};
  }

  int min_x = mask.width;
  int max_x = -1;
  int min_y = mask.height;
  int max_y = -1;
  std::vector<float> valid_depths;
  valid_depths.reserve(mask.values.size() / 8);
  for (int y = 0; y < mask.height; ++y) {
    for (int x = 0; x < mask.width; ++x) {
      const std::size_t idx = static_cast<std::size_t>(y) * mask.width + x;
      if (mask.values[idx] == 0) {
        continue;
      }
      min_x = std::min(min_x, x);
      max_x = std::max(max_x, x);
      min_y = std::min(min_y, y);
      max_y = std::max(max_y, y);
      const float z = depth_m[idx];
      if (z >= config.depth_min && z < config.depth_max) {
        valid_depths.push_back(z);
      }
    }
  }
  if (max_x < min_x || valid_depths.empty()) {
    return {0.0f, 0.0f, 0.0f};
  }
  const std::size_t middle = valid_depths.size() / 2;
  std::ranges::nth_element(valid_depths, valid_depths.begin() + middle);
  const float z = valid_depths[middle];
  const float u = (static_cast<float>(min_x) + static_cast<float>(max_x)) * 0.5f;
  const float v = (static_cast<float>(min_y) + static_cast<float>(max_y)) * 0.5f;
  return {
      (u - intrinsics.cx) * z / intrinsics.fx,
      (v - intrinsics.cy) * z / intrinsics.fy,
      z,
  };
}

std::vector<Mat4f> generateHypotheses(const std::vector<float>& depth_m,
                                      const MaskImage& mask,
                                      CameraIntrinsics intrinsics,
                                      int n_hypotheses,
                                      const Config& config) {
  std::vector<Mat3f> rotations = generateRotationGrid(config);
  if (n_hypotheses > 0 && static_cast<int>(rotations.size()) > n_hypotheses) {
    rotations.resize(static_cast<std::size_t>(n_hypotheses));
  }
  const Vec3f t = guessTranslation(depth_m, mask, intrinsics, config);
  std::vector<Mat4f> poses;
  poses.reserve(rotations.size());
  for (const Mat3f& r : rotations) {
    poses.push_back(makeTransform(r, t));
  }
  return poses;
}

std::vector<Mat3f> makeRotationGrid(const Config& config) {
  return generateRotationGrid(config);
}

std::vector<CropBox> computeCropBoxes(const std::vector<Mat4f>& poses,
                                      CameraIntrinsics intrinsics,
                                      float diameter,
                                      const Config& config) {
  const float radius = diameter * config.crop_ratio * 0.5f;
  const std::array<Vec3f, 5> offsets = {{
      {0.0f, 0.0f, 0.0f}, {radius, 0.0f, 0.0f}, {-radius, 0.0f, 0.0f},
      {0.0f, radius, 0.0f}, {0.0f, -radius, 0.0f},
  }};
  std::vector<CropBox> boxes;
  boxes.reserve(poses.size());
  for (const Mat4f& pose : poses) {
    const Vec3f center = translationPart(pose);
    std::vector<Vec2f> projected;
    projected.reserve(5);
    for (Vec3f off : offsets) {
      projected.push_back(project(center + off, intrinsics));
    }
    const Vec2f c = projected[0];
    float radius_px = 1.0f;
    for (Vec2f uv : projected) {
      radius_px = std::max(radius_px, std::abs(uv.x - c.x));
      radius_px = std::max(radius_px, std::abs(uv.y - c.y));
    }
    boxes.push_back(CropBox{
        std::round(c.x - radius_px),
        std::round(c.y - radius_px),
        std::round(c.x + radius_px),
        std::round(c.y + radius_px),
    });
  }
  return boxes;
}

Mat4f makeUncenter(Vec3f center) {
  Mat4f out = Mat4f::identity();
  out(0, 3) = -center.x;
  out(1, 3) = -center.y;
  out(2, 3) = -center.z;
  return out;
}

Mat4f applyDelta(const Mat4f& pose,
                 Vec3f delta_translation,
                 Vec3f delta_rotation,
                 float diameter,
                 const Config& config) {
  const Vec3f omega{
      std::tanh(delta_rotation.x) * config.rotation_normalizer_rad,
      std::tanh(delta_rotation.y) * config.rotation_normalizer_rad,
      std::tanh(delta_rotation.z) * config.rotation_normalizer_rad,
  };
  const Mat3f delta_r = transpose(so3Exp(omega));
  const Mat3f new_r = multiply(delta_r, rotationPart(pose));
  Vec3f new_t = translationPart(pose);
  new_t = new_t + delta_translation * (diameter * 0.5f);
  return makeTransform(new_r, new_t);
}

}  // namespace foundation_pose_nvidia::detail
