/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>

#include "foundation_pose_nvidia/config.hpp"
#include "foundation_pose_nvidia/types.hpp"

namespace foundation_pose_nvidia {

Mesh loadMesh(const std::filesystem::path& path);
void computeVertexNormals(Mesh& mesh);
PreprocessedMesh preprocessMesh(const Mesh& mesh, const Config& config);

}  // namespace foundation_pose_nvidia
