# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build for sample_app_cpp. Links the stable C ABI shared library plus libpng
# (needed for non-synthetic BOP YCB-V frame loading).
#
#   cmake -S . -B build && cmake --build build
#
# The library must be built first (repo root: docker compose run --rm build).
cmake_minimum_required(VERSION 3.16)
project(foundation_pose_sample_app_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Repo root is two levels up from example/cpp/.
get_filename_component(FP_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

find_library(FP_LIBRARY
  NAMES foundation_pose_nvidia
  HINTS "${FP_REPO_ROOT}/build"
  REQUIRED)
find_package(PNG REQUIRED)

add_executable(sample_app_cpp main.cpp frame_data.cpp)
target_include_directories(sample_app_cpp PRIVATE "${FP_REPO_ROOT}/include")
target_link_libraries(sample_app_cpp PRIVATE "${FP_LIBRARY}" PNG::PNG)
# So the loader finds libfoundation_pose_nvidia.so at runtime.
set_target_properties(sample_app_cpp PROPERTIES
  BUILD_RPATH "${FP_REPO_ROOT}/build")
