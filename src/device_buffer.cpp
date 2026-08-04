/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "foundation_pose_nvidia/device_buffer.hpp"
#include "foundation_pose_nvidia/exception.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace foundation_pose_nvidia {

void checkCuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw FoundationPoseError(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    reset();
    ptr_ = std::exchange(other.ptr_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

DeviceBuffer::~DeviceBuffer() {
  reset();
}

void DeviceBuffer::allocate(std::size_t bytes) {
  if (bytes == bytes_ && ptr_ != nullptr) {
    return;
  }
  reset();
  if (bytes == 0) {
    return;
  }
  checkCuda(cudaMalloc(&ptr_, bytes), "cudaMalloc");
  bytes_ = bytes;
}

void DeviceBuffer::reset() {
  if (ptr_ != nullptr) {
    cudaFree(ptr_);
    ptr_ = nullptr;
    bytes_ = 0;
  }
}

CudaStream::CudaStream(unsigned int flags) {
  checkCuda(cudaStreamCreateWithFlags(&stream_, flags), "cudaStreamCreateWithFlags");
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)) {}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
  if (this != &other) {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
    stream_ = std::exchange(other.stream_, nullptr);
  }
  return *this;
}

CudaStream::~CudaStream() {
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
}

void CudaStream::synchronize() const {
  checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

}  // namespace foundation_pose_nvidia
