# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from typing import Any

import numpy as np

from foundation_pose_nvidia.bindings._ctypes_types import ImageView

# Buffer requirements per fp_image_view_t field: (numpy dtype, CUDA-array-
# interface typestrs accepted for GPU inputs).
_RGB_TYPESTRS = ("|u1",)
_DEPTH_TYPESTRS = ("<f4", "=f4")
_MASK_TYPESTRS = ("|u1",)


def _cuda_interface(buffer: Any) -> dict | None:
    """Return the CUDA array interface if `buffer` is a GPU array.

    torch CUDA tensors, CuPy arrays, and Numba device arrays all expose
    `__cuda_array_interface__`, so any of them can be passed directly (e.g. a
    `torch.Tensor` on `cuda:0`). The C library accepts device pointers and
    resolves them via unified virtual addressing.
    """
    return getattr(buffer, "__cuda_array_interface__", None)


def _c_contiguous_strides(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    strides = [itemsize] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _validate_device_buffer(buffer: Any, typestrs: tuple[str, ...], ndim: int,
                            name: str) -> Any:
    """Validate dtype/contiguity of a GPU array; the wrapper never converts on
    device, so mismatches raise with a conversion hint instead."""
    cai = buffer.__cuda_array_interface__
    typestr = cai.get("typestr")
    if typestr not in typestrs:
        raise ValueError(
            f"{name}: GPU buffer has typestr {typestr!r}, expected one of {typestrs}. "
            f"Convert before passing (torch: .to(torch.float32) / .to(torch.uint8)).")
    shape = tuple(cai["shape"])
    if len(shape) != ndim:
        raise ValueError(f"{name}: GPU buffer must be {ndim}-D, got shape {shape}.")
    strides = cai.get("strides")
    itemsize = int(typestr[2:])
    if strides is not None and tuple(strides) != _c_contiguous_strides(shape, itemsize):
        raise ValueError(
            f"{name}: GPU buffer must be C-contiguous "
            f"(torch: call .contiguous() before passing).")
    return buffer


def _data_pointer(buffer: Any, ctype: type) -> Any:
    cai = _cuda_interface(buffer)
    if cai is not None:
        return C.cast(C.c_void_p(cai["data"][0]), C.POINTER(ctype))
    return buffer.ctypes.data_as(C.POINTER(ctype))


@dataclass
class RgbdFrame:
    """One RGB-D frame in the layout expected by fp_image_view_t.

    `rgb`, `depth_m`, and `mask` may each be a host numpy array OR a GPU array
    exposing `__cuda_array_interface__` (torch CUDA tensor, CuPy, Numba); GPU
    inputs skip the host-to-device copy and may be mixed with host inputs.
    GPU buffers must already be C-contiguous with the exact dtype (uint8 RGB /
    mask, float32 depth in meters) — they are validated, never converted.
    `intrinsics` is always host memory.

    Stream ordering: the estimator issues a device synchronize before reading
    the frame, so tensors produced asynchronously (e.g. by torch kernels on
    another stream) are safe to pass without an explicit torch.cuda.synchronize().
    """

    rgb: Any
    depth_m: Any
    intrinsics: np.ndarray
    mask: Any | None = None

    def __post_init__(self) -> None:
        if _cuda_interface(self.rgb) is not None:
            _validate_device_buffer(self.rgb, _RGB_TYPESTRS, 3, "rgb")
        else:
            self.rgb = np.ascontiguousarray(self.rgb, dtype=np.uint8)
        if _cuda_interface(self.depth_m) is not None:
            _validate_device_buffer(self.depth_m, _DEPTH_TYPESTRS, 2, "depth_m")
        else:
            self.depth_m = np.ascontiguousarray(self.depth_m, dtype=np.float32)
        if _cuda_interface(self.intrinsics) is not None:
            raise ValueError("intrinsics must be host memory (numpy), not a GPU array")
        self.intrinsics = np.ascontiguousarray(
            np.asarray(self.intrinsics).reshape(3, 3), dtype=np.float32)
        if self.mask is not None:
            if _cuda_interface(self.mask) is not None:
                _validate_device_buffer(self.mask, _MASK_TYPESTRS, 2, "mask")
            else:
                self.mask = np.ascontiguousarray(self.mask, dtype=np.uint8)

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    def to_image_view(self, *, with_mask: bool) -> ImageView:
        mask = self.mask if with_mask else None
        if with_mask and mask is None:
            raise ValueError("register requires a mask on RgbdFrame")

        rgb = self.rgb
        depth_m = self.depth_m
        k = np.ascontiguousarray(self.intrinsics.reshape(9), dtype=np.float32)

        mask_ptr = C.POINTER(C.c_uint8)()
        if mask is not None:
            mask_ptr = _data_pointer(mask, C.c_uint8)

        view = ImageView(
            self.width,
            self.height,
            _data_pointer(rgb, C.c_uint8),
            _data_pointer(depth_m, C.c_float),
            mask_ptr,
            k.ctypes.data_as(C.POINTER(C.c_float)),
        )
        # Pin buffers to the view: ctypes stores only raw addresses.
        view._keepalive = (rgb, depth_m, mask, k)
        return view
