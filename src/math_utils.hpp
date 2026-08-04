/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "foundation_pose_nvidia/types.hpp"

namespace foundation_pose_nvidia::detail {

constexpr float kPi = std::numbers::pi_v<float>;

// Vec3f arithmetic operators are hidden friends of Vec3f (see types.hpp); they
// are found here via argument-dependent lookup.

inline float dot(Vec3f a, Vec3f b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3f cross(Vec3f a, Vec3f b) {
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

inline float norm(Vec3f a) {
  return std::sqrt(std::max(dot(a, a), 0.0f));
}

inline Vec3f normalize(Vec3f a) {
  const float n = norm(a);
  if (n < 1e-12f) {
    return {0.0f, 0.0f, 0.0f};
  }
  return a / n;
}

inline Mat3f transpose(const Mat3f& m) {
  Mat3f out;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out(r, c) = m(c, r);
    }
  }
  return out;
}

inline Mat3f multiply(const Mat3f& a, const Mat3f& b) {
  Mat3f out;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      float v = 0.0f;
      for (int k = 0; k < 3; ++k) {
        v += a(r, k) * b(k, c);
      }
      out(r, c) = v;
    }
  }
  return out;
}

inline Vec3f multiply(const Mat3f& m, Vec3f p) {
  return {
      m(0, 0) * p.x + m(0, 1) * p.y + m(0, 2) * p.z,
      m(1, 0) * p.x + m(1, 1) * p.y + m(1, 2) * p.z,
      m(2, 0) * p.x + m(2, 1) * p.y + m(2, 2) * p.z,
  };
}

inline Mat4f multiply(const Mat4f& a, const Mat4f& b) {
  Mat4f out;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      float v = 0.0f;
      for (int k = 0; k < 4; ++k) {
        v += a(r, k) * b(k, c);
      }
      out(r, c) = v;
    }
  }
  return out;
}

inline Vec3f transformPoint(const Mat4f& t, Vec3f p) {
  const float x = t(0, 0) * p.x + t(0, 1) * p.y + t(0, 2) * p.z + t(0, 3);
  const float y = t(1, 0) * p.x + t(1, 1) * p.y + t(1, 2) * p.z + t(1, 3);
  const float z = t(2, 0) * p.x + t(2, 1) * p.y + t(2, 2) * p.z + t(2, 3);
  if (const float w = t(3, 0) * p.x + t(3, 1) * p.y + t(3, 2) * p.z + t(3, 3);
      std::abs(w) > 1e-12f && std::abs(w - 1.0f) > 1e-6f) {
    return {x / w, y / w, z / w};
  }
  return {x, y, z};
}

inline Mat3f rotationPart(const Mat4f& t) {
  Mat3f out;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out(r, c) = t(r, c);
    }
  }
  return out;
}

inline void setRotationPart(Mat4f& t, const Mat3f& r) {
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      t(row, col) = r(row, col);
    }
  }
}

inline Vec3f translationPart(const Mat4f& t) {
  return {t(0, 3), t(1, 3), t(2, 3)};
}

inline void setTranslationPart(Mat4f& t, Vec3f p) {
  t(0, 3) = p.x;
  t(1, 3) = p.y;
  t(2, 3) = p.z;
}

inline Mat4f makeTransform(const Mat3f& r, Vec3f t) {
  Mat4f out = Mat4f::identity();
  setRotationPart(out, r);
  setTranslationPart(out, t);
  return out;
}

inline Mat4f inverseRigid(const Mat4f& t) {
  const Mat3f r_t = transpose(rotationPart(t));
  const Vec3f tr = translationPart(t);
  const Vec3f inv_t = multiply(r_t, tr) * -1.0f;
  return makeTransform(r_t, inv_t);
}

inline Mat3f rotationZ(float theta) {
  Mat3f out = Mat3f::identity();
  const float c = std::cos(theta);
  const float s = std::sin(theta);
  out(0, 0) = c;
  out(0, 1) = -s;
  out(1, 0) = s;
  out(1, 1) = c;
  return out;
}

inline Mat3f so3Exp(Vec3f omega) {
  const float theta = norm(omega);
  if (theta < 1e-6f) {
    return Mat3f::identity();
  }
  const Vec3f k = omega / theta;
  Mat3f kk;
  kk(0, 1) = -k.z;
  kk(0, 2) = k.y;
  kk(1, 0) = k.z;
  kk(1, 2) = -k.x;
  kk(2, 0) = -k.y;
  kk(2, 1) = k.x;
  const Mat3f kk2 = multiply(kk, kk);
  Mat3f out = Mat3f::identity();
  const float s = std::sin(theta);
  const float c = std::cos(theta);
  for (int r = 0; r < 3; ++r) {
    for (int col = 0; col < 3; ++col) {
      out(r, col) += s * kk(r, col) + (1.0f - c) * kk2(r, col);
    }
  }
  return out;
}

inline float geodesicDistance(const Mat3f& a, const Mat3f& b) {
  const Mat3f rel = multiply(a, transpose(b));
  const float trace = rel(0, 0) + rel(1, 1) + rel(2, 2);
  const float cos_theta = std::clamp((trace - 1.0f) * 0.5f, -1.0f, 1.0f);
  return std::acos(cos_theta);
}

inline Vec2f project(Vec3f p, CameraIntrinsics k) {
  if (std::abs(p.z) < 1e-12f) {
    return {std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN()};
  }
  return {k.fx * p.x / p.z + k.cx, k.fy * p.y / p.z + k.cy};
}

inline float edgeFunction(Vec2f a, Vec2f b, Vec2f c) {
  return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

}  // namespace foundation_pose_nvidia::detail
