// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PNG loading helpers for sample_app_cpp. Used in non-synthetic mode to decode
// BOP YCB-V RGB/depth/mask PNGs.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sample {

// Load an 8-bit RGB PNG as HWC (width*height*3 bytes). Throws on failure.
std::vector<uint8_t> loadRgb(const std::string& path, int& width, int& height);

// Load a 16-bit grayscale depth PNG and convert to meters:
//   meters = raw_uint16 * depth_scale / 1000.
std::vector<float> loadDepthMeters(const std::string& path, float depth_scale,
                                   int& width, int& height);

// Load an 8-bit grayscale mask PNG (non-zero = object). Throws on failure.
std::vector<uint8_t> loadMask(const std::string& path, int& width, int& height);

}  // namespace sample
