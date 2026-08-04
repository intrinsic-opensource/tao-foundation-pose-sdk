/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace foundation_pose_nvidia {

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes) { allocate(bytes); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
  ~DeviceBuffer();

  void allocate(std::size_t bytes);
  void reset();

  void* data() { return ptr_; }
  const void* data() const { return ptr_; }
  std::size_t bytes() const { return bytes_; }

  template <class T>
  T* as() {
    return static_cast<T*>(ptr_);
  }

  template <class T>
  const T* as() const {
    return static_cast<const T*>(ptr_);
  }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

class CudaStream {
 public:
  explicit CudaStream(unsigned int flags = cudaStreamNonBlocking);
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  CudaStream(CudaStream&& other) noexcept;
  CudaStream& operator=(CudaStream&& other) noexcept;
  ~CudaStream();

  cudaStream_t get() const { return stream_; }
  void synchronize() const;

 private:
  cudaStream_t stream_ = nullptr;
};

void checkCuda(cudaError_t status, const char* what);

}  // namespace foundation_pose_nvidia
