/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdexcept>

namespace foundation_pose_nvidia {

// Dedicated exception type for all runtime error conditions raised by the
// FoundationPose runtime. It derives from std::runtime_error so every existing
// `catch (const std::exception&)` / `catch (const std::runtime_error&)` handler
// (including the ones guarding the C ABI boundary in c_api.cpp) keeps working
// unchanged, while giving callers a single, greppable type to catch when they
// want to distinguish FoundationPose failures from other std exceptions.
class FoundationPoseError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace foundation_pose_nvidia
