# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes as C
import gc
import os
import subprocess
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Make sure the package is importable from the source tree without an install.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "src"))

from foundation_pose_nvidia.bindings._ctypes_types import Config, ImageView
from foundation_pose_nvidia.core.estimator import Estimator
from foundation_pose_nvidia.core.frame import RgbdFrame
from foundation_pose_nvidia.core.options import EstimatorOptions
from foundation_pose_nvidia.exceptions import FoundationPoseError


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

# A non-zero integer that serves as the "fake C handle" returned by fp_create.
FAKE_HANDLE_VALUE = 0xDEAD_C0DE


def _make_fake_cdll() -> MagicMock:
    """Return a MagicMock that mimics the raw ctypes CDLL for the C library."""
    lib = MagicMock()
    lib.fp_create.return_value = FAKE_HANDLE_VALUE
    lib.fp_prepare.return_value = 0        # 0 = success
    lib.fp_destroy.return_value = None
    lib.fp_register_frame.return_value = 0
    lib.fp_track_frame.return_value = 0
    lib.fp_synchronize_device.return_value = 0
    lib.fp_build_info.return_value = b"fake-build-info"
    return lib


def _make_fake_library(cdll: MagicMock) -> MagicMock:
    """Return a MagicMock that mimics an FPLibrary wrapping the given cdll mock."""
    library = MagicMock()
    library.cdll = cdll
    library.default_config.return_value = Config()
    library.synchronize.return_value = None
    return library


def _make_options(prepare: bool = False) -> EstimatorOptions:
    return EstimatorOptions(
        cad_path=Path("/fake/model.obj"),
        refine_model_path=Path("/fake/refiner.onnx"),
        score_model_path=Path("/fake/scorer.onnx"),
        engine_cache_dir=Path("/fake/cache"),
        prepare=prepare,
    )


def _build_estimator(
    cdll: MagicMock,
    library: MagicMock,
    *,
    options: EstimatorOptions | None = None,
    prepare_batch: int | None = None,
    prepare_track_batch: bool = False,
) -> Estimator:
    """Construct a fully-mocked Estimator."""
    if options is None:
        options = _make_options()
    return Estimator(
        options,
        Config(),
        library=library,
        prepare_batch=prepare_batch,
        prepare_track_batch=prepare_track_batch,
    )


def _make_arrays(
    h: int = 120,
    w: int = 160,
    depth_dtype: type = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (rgb, depth, mask, K) arrays of the given shape."""
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    depth = np.zeros((h, w), dtype=depth_dtype)
    mask = np.ones((h, w), dtype=np.uint8)
    K = np.array([[500, 0, w / 2], [0, 500, h / 2], [0, 0, 1]], dtype=np.float32)
    return rgb, depth, mask, K


# ===========================================================================
# C handle leaks when _prepare raises after fp_create
# ===========================================================================

class TestBugExceptionSafetyHandleLeak(unittest.TestCase):
    """

    Make sure esimator destroyer is called in the wrapper if there is any failure during creating / preparing the estimator
    """

    def _configure_prepare_to_fail_on(
        self, cdll: MagicMock, fail_on_nth_call: int
    ) -> None:
        """Make fp_prepare return 1 (failure) on the Nth invocation."""
        call_counter: list[int] = [0]

        def side_effect(*_args, **_kwargs):
            call_counter[0] += 1
            return 1 if call_counter[0] == fail_on_nth_call else 0

        cdll.fp_prepare.side_effect = side_effect

    def test_bug1_first_prepare_fails_handle_must_be_destroyed(self):
        """fp_destroy must be called when the first _prepare call raises.
        """
        cdll = _make_fake_cdll()
        library = _make_fake_library(cdll)
        self._configure_prepare_to_fail_on(cdll, fail_on_nth_call=1)

        options = _make_options(prepare=True)

        with self.assertRaises(FoundationPoseError):
            _build_estimator(cdll, library, options=options, prepare_batch=252)

        # Test passes if the fp_destroy has been called at least 1 when there is assertion when preparing the estimator.
        cdll.fp_destroy.assert_called_once_with(FAKE_HANDLE_VALUE)

    def test_bug1_prepare_track_batch_fails_handle_must_be_destroyed(self):
        """fp_destroy must be called when _prepare call raises if prepare_track_batch is used
        """
        cdll = _make_fake_cdll()
        library = _make_fake_library(cdll)
        self._configure_prepare_to_fail_on(cdll, fail_on_nth_call=1)

        options = _make_options(prepare=True)

        with self.assertRaises(FoundationPoseError):
            _build_estimator(
                cdll,
                library,
                options=options,
                prepare_batch=252,
                prepare_track_batch=True,
            )

        # Test passes if the fp_destroy has been called at least 1 when there is assertion when preparing the estimator.
        cdll.fp_destroy.assert_called_once_with(FAKE_HANDLE_VALUE)


# ===========================================================================
# Buffer lifetime: copies owned by anonymous RgbdFrame must survive until C call
# ===========================================================================

class TestBugRgbdFrameTemporaryFreedBeforeCCall(unittest.TestCase):
    """Verify that buffer arrays pinned by view._keepalive are alive at C call time.
    """

    def _make_capturing_frame_class(
        self, depth_weakrefs: list, rgb_weakrefs: list
    ) -> type:
        """Return a RgbdFrame subclass that records weakrefs to its buffer copies."""

        class CapturingRgbdFrame(RgbdFrame):
            def __post_init__(self) -> None:
                super().__post_init__()
                depth_weakrefs.append(weakref.ref(self.depth_m))
                rgb_weakrefs.append(weakref.ref(self.rgb))

        return CapturingRgbdFrame

    def test_register_arrays_depth_buffer_alive_at_c_call(self):
        """view._keepalive must keep the float32 depth copy alive at fp_register_frame.

        A float64 depth triggers a float32 copy in __post_init__.  Without the
        fix the copy is freed when the anonymous RgbdFrame is collected, leaving
        a dangling pointer in the ImageView.  With the fix, view._keepalive holds
        the copy so it outlives the frame.
        """
        depth_weakrefs: list = []
        rgb_weakrefs: list = []
        buffer_alive_at_c_call: list[bool] = []

        cdll = _make_fake_cdll()
        library = _make_fake_library(cdll)
        CapturingRgbdFrame = self._make_capturing_frame_class(
            depth_weakrefs, rgb_weakrefs
        )

        def mock_fp_register_frame(*args, **kwargs):
            gc.collect()
            buffer_alive_at_c_call.append(depth_weakrefs[-1]() is not None)
            return 0

        cdll.fp_register_frame.side_effect = mock_fp_register_frame

        """ 
        Wrapping the RgbdFrame used in source code with this CapturingRgbdFrame, in order to add mocking materials for test
        """
        
        with patch(
            "foundation_pose_nvidia.core.estimator.RgbdFrame",
            CapturingRgbdFrame,
        ):
            estimator = _build_estimator(cdll, library)
            rgb, depth_f64, mask, K = _make_arrays(depth_dtype=np.float64)
            estimator._register_arrays(rgb, depth_f64, mask, K, -1, -1)

        self.assertTrue(
            buffer_alive_at_c_call,
            "fp_register_frame side-effect was never invoked; test setup is wrong.",
        )
        self.assertTrue(
            buffer_alive_at_c_call[0],
            "float32 depth copy was freed BEFORE fp_register_frame was called. "
            "view._keepalive must pin it to the ImageView.",
        )

    def test_track_tuple_depth_buffer_alive_at_c_call(self):
        """view._keepalive must keep the depth copy alive at fp_track_frame.

        estimator.py track() tuple path creates an anonymous RgbdFrame:
            view = RgbdFrame(rgb, depth_m, k).to_image_view(with_mask=False)
        Same hazard as _register_arrays.
        """
        depth_weakrefs: list = []
        rgb_weakrefs: list = []
        buffer_alive_at_c_call: list[bool] = []

        cdll = _make_fake_cdll()
        library = _make_fake_library(cdll)
        CapturingRgbdFrame = self._make_capturing_frame_class(
            depth_weakrefs, rgb_weakrefs
        )

        def mock_fp_track_frame(*args, **kwargs):
            gc.collect()
            buffer_alive_at_c_call.append(depth_weakrefs[-1]() is not None)
            return 0

        cdll.fp_track_frame.side_effect = mock_fp_track_frame

        with patch(
            "foundation_pose_nvidia.core.estimator.RgbdFrame",
            CapturingRgbdFrame,
        ):
            estimator = _build_estimator(cdll, library)
            rgb, depth_f64, _mask, K = _make_arrays(depth_dtype=np.float64)
            estimator.track((rgb, depth_f64, K), n_refine=-1)

        self.assertTrue(
            buffer_alive_at_c_call,
            "fp_track_frame side-effect was never invoked; test setup is wrong.",
        )
        self.assertTrue(
            buffer_alive_at_c_call[0],
            "float32 depth copy was freed BEFORE fp_track_frame was called. "
            "view._keepalive must pin it to the ImageView.",
        )

    def test_non_contiguous_rgb_buffer_alive_at_c_call(self):
        """view._keepalive must keep the contiguous rgb copy alive at fp_register_frame.

        A non-contiguous rgb slice (arr[::2]) triggers a contiguous copy in
        __post_init__.  That copy must survive until the C call via view._keepalive.
        """
        depth_weakrefs: list = []
        rgb_weakrefs: list = []
        buffer_alive_at_c_call: list[bool] = []

        cdll = _make_fake_cdll()
        library = _make_fake_library(cdll)
        CapturingRgbdFrame = self._make_capturing_frame_class(
            depth_weakrefs, rgb_weakrefs
        )

        def mock_fp_register_frame(*args, **kwargs):
            gc.collect()
            buffer_alive_at_c_call.append(rgb_weakrefs[-1]() is not None)
            return 0

        cdll.fp_register_frame.side_effect = mock_fp_register_frame

        with patch(
            "foundation_pose_nvidia.core.estimator.RgbdFrame",
            CapturingRgbdFrame,
        ):
            estimator = _build_estimator(cdll, library)
            rgb, depth, mask, K = _make_arrays()
            rgb_nc = rgb[::2]
            depth_nc = depth[::2]
            mask_nc = mask[::2]
            estimator._register_arrays(rgb_nc, depth_nc, mask_nc, K, -1, -1)

        self.assertTrue(
            buffer_alive_at_c_call,
            "fp_register_frame side-effect was never invoked; test setup is wrong.",
        )
        self.assertTrue(
            buffer_alive_at_c_call[0],
            "contiguous rgb copy was freed BEFORE fp_register_frame was called. "
            "view._keepalive must pin it to the ImageView.",
        )

    def test_track_arrays_public_api_depth_buffer_alive_at_c_call(self):
        """Public track_arrays() must keep the depth copy alive at fp_track_frame.

        track_arrays() → track((rgb, depth, K)) creates an anonymous RgbdFrame
        internally.  Exercises the user-facing API surface.
        """
        depth_weakrefs: list = []
        rgb_weakrefs: list = []
        buffer_alive_at_c_call: list[bool] = []

        cdll = _make_fake_cdll()
        library = _make_fake_library(cdll)
        CapturingRgbdFrame = self._make_capturing_frame_class(
            depth_weakrefs, rgb_weakrefs
        )

        def mock_fp_track_frame(*args, **kwargs):
            gc.collect()
            buffer_alive_at_c_call.append(depth_weakrefs[-1]() is not None)
            return 0

        cdll.fp_track_frame.side_effect = mock_fp_track_frame

        with patch(
            "foundation_pose_nvidia.core.estimator.RgbdFrame",
            CapturingRgbdFrame,
        ):
            estimator = _build_estimator(cdll, library)
            rgb, depth_f64, _mask, K = _make_arrays(depth_dtype=np.float64)
            estimator.track_arrays(rgb, depth_f64, K, n_refine=-1)

        self.assertTrue(
            buffer_alive_at_c_call,
            "fp_track_frame side-effect was never invoked; test setup is wrong.",
        )
        self.assertTrue(
            buffer_alive_at_c_call[0],
            "float32 depth copy was freed BEFORE fp_track_frame was called via "
            "track_arrays().  view._keepalive must pin it to the ImageView.",
        )


# ===========================================================================
# Buffer lifetime: local 'k' array in to_image_view not anchored
# ===========================================================================

class TestBug3LocalKArrayNotAnchoredToImageView(unittest.TestCase):
    
    def test_bug3_latent_safety_k_is_currently_a_view_of_intrinsics(self):
        """Document current safety: k shares the self.intrinsics data buffer.

        Because __post_init__ stores self.intrinsics as contiguous float32,
        np.ascontiguousarray returns a view (zero-copy).  The pointer in
        k_row_major is therefore backed by self.intrinsics, which lives as long
        as the RgbdFrame.

        This test PASSES on current code.  It will FAIL if __post_init__ is
        ever changed so that self.intrinsics is no longer contiguous float32,
        which would make to_image_view silently hand out a dangling pointer.
        """
        _, depth, mask, K = _make_arrays()
        rgb = np.full((120, 160, 3), 64, dtype=np.uint8)
        frame = RgbdFrame(rgb, depth, K, mask)

        # Simulate what to_image_view does internally:
        k = np.ascontiguousarray(frame.intrinsics.reshape(9), dtype=np.float32)

        self.assertTrue(
            np.shares_memory(k, frame.intrinsics),
            "LATENT BUG #3 IS NOW ACTIVE: k is a COPY, not a view of "
            "self.intrinsics.  to_image_view now hands out a dangling pointer "
            "to the freed 'k' copy after the function returns.",
        )

    def test_bug3_image_view_holds_no_python_ref_to_k(self):
        """ImageView stores only the raw pointer to k, not a Python reference.

        This structural test confirms the root cause: there is no mechanism in
        the current code to keep k alive via the ImageView.  The ctypes
        Structure stores the integer pointer value; it has no slots for Python
        object references.
        """
        _, depth, mask, K = _make_arrays()
        rgb = np.full((120, 160, 3), 64, dtype=np.uint8)
        frame = RgbdFrame(rgb, depth, K, mask)
        view = frame.to_image_view(with_mask=True)

        # A ctypes Structure only references Python objects via its _objects
        # internal dict (for keeping alive c_char_p strings etc).
        # numpy arrays are never in _objects because they provide raw buffers
        # via __array_interface__, not ctypes-managed memory.
        k_ptr_int = C.cast(view.k_row_major, C.c_void_p).value
        self.assertIsNotNone(k_ptr_int, "k_row_major pointer should not be null")

        # Confirm no reference to frame.intrinsics is embedded *inside* the view.
        # gc.get_referents() returns the Python objects that `view` holds references
        # TO (outgoing edges).  gc.get_referrers() is the reverse direction and would
        # tell us what holds a reference to the view — which is wrong for this check.
        # Use identity (`is`) instead of equality (`==`) to avoid numpy's
        # element-wise truth-value ambiguity when comparing arrays.
        referents_of_view = gc.get_referents(view)
        self.assertFalse(
            any(obj is frame.intrinsics for obj in referents_of_view),
            "Unexpected: ImageView holds a Python reference to frame.intrinsics. "
            "This would actually be the CORRECT fix — it would keep the buffer alive.",
        )

    def test_bug3_copy_of_k_freed_immediately_on_scope_exit(self):
        """Demonstrate that if k were a copy it would be freed before the C call.

        We use a weakref to a copy of the k array to observe when CPython would
        free it.  The copy goes out of scope (and is freed in CPython) as soon as
        to_image_view returns — the ImageView has no reference to extend k's life.

        This test PASSES on current code (k IS freed immediately when it's a copy)
        and thereby documents why the invariant "k must be a view" is critical.
        """
        _, depth, mask, K = _make_arrays()
        rgb = np.full((120, 160, 3), 64, dtype=np.uint8)
        frame = RgbdFrame(rgb, depth, K, mask)

        # Simulate making a copy — the scenario that would fire the bug:
        k_copy = frame.intrinsics.reshape(9).astype(np.float32, copy=True)
        ref = weakref.ref(k_copy)
        self.assertFalse(
            np.shares_memory(k_copy, frame.intrinsics),
            "k_copy should be independent (a real copy, not a view).",
        )

        # Drop the only Python reference (mirroring what happens inside
        # to_image_view when k goes out of scope on return).
        del k_copy
        gc.collect()

        # In CPython the copy is freed immediately; weakref becomes None.
        self.assertIsNone(
            ref(),
            "LATENT BUG #3: a copy of k is freed as soon as to_image_view "
            "returns.  The k_row_major pointer in the ImageView would be "
            "dangling at the time of any C call that dereferences it.",
        )


# ===========================================================================
# Memory leak: no __del__ on Estimator
# ===========================================================================

class TestBugNoDelFinalizer(unittest.TestCase):
    """
    Estimator provides close() and __enter__/__exit__ for releasing the C
    handle but does NOT implement __del__.  If a caller constructs an Estimator
    without a with-statement and forgets to call close() (the natural pattern
    when assigning to a variable), fp_destroy is never invoked.

    The fp_handle heap object plus the FoundationPose object it owns
    (TensorRT engines, CUDA stream, GPU workspace, uploaded mesh) are leaked
    for the process lifetime.

    Fix: add __del__ that calls self.close() (or equivalently, calls
    fp_destroy directly if self.handle is set).

    test_bug4_del_estimator_triggers_fp_destroy SHOULD FAIL on the current
    (unpatched) code, confirming the bug exists.
    """

    def test_bug_del_estimator_triggers_fp_destroy(self):
        """Deleting an Estimator (without close() or with-statement) must call fp_destroy.
        """
        cdll = _make_fake_cdll()
        library = _make_fake_library(cdll)

        estimator = _build_estimator(cdll, library)
        # Simulates a caller that forgets close() / with-statement:
        del estimator
        gc.collect()

        cdll.fp_destroy.assert_called_once_with(FAKE_HANDLE_VALUE)

    def test_bug_close_explicitly_still_works(self):
        """Calling close() explicitly must always invoke fp_destroy (control case).
        """
        cdll = _make_fake_cdll()
        library = _make_fake_library(cdll)

        estimator = _build_estimator(cdll, library)
        estimator.close()
        gc.collect()

        cdll.fp_destroy.assert_called_once_with(FAKE_HANDLE_VALUE)

    def test_bug_context_manager_calls_fp_destroy(self):
        """Using Estimator as a context manager must invoke fp_destroy (control case).
        """
        cdll = _make_fake_cdll()
        library = _make_fake_library(cdll)

        with _build_estimator(cdll, library):
            pass

        cdll.fp_destroy.assert_called_once_with(FAKE_HANDLE_VALUE)

    def test_bug_multiple_estimators_without_close_all_leak(self):
        """N Estimators without close() produce N leaked handles (confirms the scale).
        """
        N = 5
        cdll = _make_fake_cdll()
        library = _make_fake_library(cdll)

        estimators = [_build_estimator(cdll, library) for _ in range(N)]
        del estimators
        gc.collect()

        destroy_count = cdll.fp_destroy.call_count
        self.assertEqual(
            destroy_count,
            N,
            f"BUG #4: expected fp_destroy called {N} times, got {destroy_count}. "
            f"{N - destroy_count} C handle(s) leaked.",
        )

    def test_bug_double_destroy_not_called_when_close_then_del(self):
        """Calling close() then letting the Estimator be GC'd must NOT double-destroy.
        """
        cdll = _make_fake_cdll()
        library = _make_fake_library(cdll)

        estimator = _build_estimator(cdll, library)
        estimator.close()                   # first explicit release
        # Now self.handle is None; a __del__-based close() call must be a no-op.
        del estimator
        gc.collect()

        # fp_destroy must have been called exactly once (by close()), not twice.
        cdll.fp_destroy.assert_called_once_with(FAKE_HANDLE_VALUE)


# ===========================================================================
# Helpers for sanity / VRAM tests
# ===========================================================================

def _nvidia_smi_vram_mb(gpu_index: int = 0) -> int | None:
    """Return VRAM used in MiB for the given GPU via nvidia-smi, or None."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        return int(out.decode().strip())
    except Exception:
        return None


def _synthesize_rgbd(
    h: int = 120, w: int = 160
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (rgb, depth, mask, K) with a Gaussian blob simulating a near object."""
    rng = np.random.default_rng(seed=0)
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = (
        0.6 + 0.2 * np.exp(-((x - w / 2) ** 2 + (y - h / 2) ** 2) / (2 * (w / 8) ** 2))
    ).astype(np.float32)
    mask = (depth > 0.65).astype(np.uint8)
    K = np.array([[500, 0, w / 2], [0, 500, h / 2], [0, 0, 1]], dtype=np.float32)
    return rgb, depth, mask, K


# Minimal watertight unit-cube OBJ — suitable as a synthesized CAD model.
_CUBE_OBJ = """\
# synthesized unit cube
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""

_GPU_AVAILABLE: bool = _nvidia_smi_vram_mb() is not None


def _try_load_real_library() -> Any:
    """Load the real C library and return it, or return None on failure."""
    try:
        from foundation_pose_nvidia.bindings import load_library  # type: ignore[attr-defined]
        return load_library(print_build_info=False)
    except Exception:
        return None


_REAL_LIB: Any = _try_load_real_library()


# ===========================================================================
# Sanity: 2-cycle create / register / track / destroy with synthesized data
# ===========================================================================

class TestSanityTwoCycleLifecycle(unittest.TestCase):
    """2 full create → register → track → destroy cycles with synthesized RGB-D data.

    Uses a mocked C library so no GPU or real model files are required.
    Each cycle asserts the correct C call sequence and that fp_destroy is
    called exactly once — the mock-level proxy for "no VRAM leak".

    For actual VRAM measurement see TestSanityVRAMLeak.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.rgb, cls.depth, cls.mask, cls.K = _synthesize_rgbd()

    def _run_one_cycle(self, cdll: MagicMock, library: MagicMock) -> None:
        frame = RgbdFrame(self.rgb, self.depth, self.K, self.mask)
        with _build_estimator(cdll, library) as est:
            est.register(frame, n_refine=2, n_hypotheses=8)
            est.track(frame, n_refine=2)

    def test_two_cycles_context_manager(self):
        """Each cycle via context manager must call fp_create / register / track /
        destroy exactly once with no leftover handles."""
        for cycle in range(2):
            with self.subTest(cycle=cycle):
                cdll = _make_fake_cdll()
                library = _make_fake_library(cdll)
                self._run_one_cycle(cdll, library)

                cdll.fp_create.assert_called_once()
                cdll.fp_register_frame.assert_called_once()
                cdll.fp_track_frame.assert_called_once()
                cdll.fp_destroy.assert_called_once_with(FAKE_HANDLE_VALUE)

    def test_two_cycles_explicit_close(self):
        """Each cycle via explicit close() must release the handle exactly once."""
        for cycle in range(2):
            with self.subTest(cycle=cycle):
                cdll = _make_fake_cdll()
                library = _make_fake_library(cdll)
                frame = RgbdFrame(self.rgb, self.depth, self.K, self.mask)
                est = _build_estimator(cdll, library)
                est.register(frame, n_refine=2, n_hypotheses=8)
                est.track(frame, n_refine=2)
                est.close()
                gc.collect()

                cdll.fp_destroy.assert_called_once_with(FAKE_HANDLE_VALUE)

    def test_two_cycles_gc_del(self):
        """Each cycle via GC (no close/with) must release the handle via __del__."""
        for cycle in range(2):
            with self.subTest(cycle=cycle):
                cdll = _make_fake_cdll()
                library = _make_fake_library(cdll)
                frame = RgbdFrame(self.rgb, self.depth, self.K, self.mask)
                est = _build_estimator(cdll, library)
                est.register(frame, n_refine=2, n_hypotheses=8)
                est.track(frame, n_refine=2)
                del est
                gc.collect()

                cdll.fp_destroy.assert_called_once_with(FAKE_HANDLE_VALUE)

    def test_two_cycles_float64_depth(self):
        """Float64 depth (triggers a buffer copy) must not affect lifecycle correctness."""
        _, depth_f64, _, _ = _make_arrays(depth_dtype=np.float64)
        for cycle in range(2):
            with self.subTest(cycle=cycle):
                cdll = _make_fake_cdll()
                library = _make_fake_library(cdll)
                frame = RgbdFrame(self.rgb, depth_f64, self.K, self.mask)
                with _build_estimator(cdll, library) as est:
                    est.register(frame, n_refine=2, n_hypotheses=8)
                    est.track(frame, n_refine=2)

                cdll.fp_destroy.assert_called_once_with(FAKE_HANDLE_VALUE)


# ===========================================================================
# Sanity: VRAM leak detection against a real GPU
# ===========================================================================

@unittest.skipUnless(_GPU_AVAILABLE, "requires NVIDIA GPU (nvidia-smi not found)")
class TestSanityVRAMLeak(unittest.TestCase):
    """VRAM leak detection: 2 cycles against the real C library.

    Requires all of:
      - NVIDIA GPU accessible via nvidia-smi  (checked by class decorator)
      - Real C library loadable by load_library()
      - Model files pointed to by env var FOUNDATION_POSE_TEST_MODEL_DIR:
            <dir>/refiner.onnx
            <dir>/scorer.onnx
            <dir>/cache/   (engine cache, may be empty on first run)

    The synthesized unit-cube OBJ is written to a temp file automatically.
    VRAM tolerance is 32 MiB to absorb driver-level rounding; a real leak
    (mesh + TensorRT engines) would typically be hundreds of MiB.
    """

    _library: Any = None
    _tmp_cad: Any = None
    _model_dir: Path | None = None

    VRAM_TOLERANCE_MB: int = 32
    GPU_INDEX: int = 0

    @classmethod
    def setUpClass(cls) -> None:
        if _REAL_LIB is None:
            raise unittest.SkipTest("Real C library not loadable (libfoundation_pose.so missing?)")

        model_dir_env = os.environ.get("FOUNDATION_POSE_TEST_MODEL_DIR")
        if not model_dir_env:
            raise unittest.SkipTest(
                "FOUNDATION_POSE_TEST_MODEL_DIR not set; "
                "point it to a directory containing refiner.onnx, scorer.onnx, cache/"
            )
        cls._model_dir = Path(model_dir_env)
        cls._library = _REAL_LIB

        # Write the synthesized CAD so we don't need a real mesh on disk.
        tmp = tempfile.NamedTemporaryFile(suffix=".obj", mode="w", delete=False)
        tmp.write(_CUBE_OBJ)
        tmp.flush()
        cls._tmp_cad = tmp

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._tmp_cad is not None:
            try:
                os.unlink(cls._tmp_cad.name)
            except OSError:
                pass

    def _make_real_options(self) -> EstimatorOptions:
        return EstimatorOptions(
            cad_path=Path(self._tmp_cad.name),
            refine_model_path=self._model_dir / "refiner.onnx",
            score_model_path=self._model_dir / "scorer.onnx",
            engine_cache_dir=self._model_dir / "cache",
            prepare=False,
        )

    def _run_real_cycle(self, rgb: np.ndarray, depth: np.ndarray,
                        mask: np.ndarray, K: np.ndarray) -> None:
        options = self._make_real_options()
        with Estimator(options, self._library.default_config(),
                       library=self._library) as est:
            frame = RgbdFrame(rgb, depth, K, mask)
            est.register(frame, n_refine=1, n_hypotheses=4)
            est.track(frame, n_refine=1)

    def test_two_cycles_vram_stable(self) -> None:
        """VRAM after 2 destroy() calls must be within tolerance of baseline.

        A leaked handle retains the TensorRT engines and GPU workspace for
        the process lifetime; the VRAM delta would be proportional to model
        size (typically hundreds of MiB), far above the 32 MiB tolerance.
        """
        rgb, depth, mask, K = _synthesize_rgbd()

        # Warm-up: establish the CUDA context so it doesn't count as a leak.
        self._run_real_cycle(rgb, depth, mask, K)

        baseline_mb = _nvidia_smi_vram_mb(self.GPU_INDEX)
        self.assertIsNotNone(baseline_mb, "nvidia-smi unavailable mid-test")

        for _ in range(2):
            self._run_real_cycle(rgb, depth, mask, K)

        after_mb = _nvidia_smi_vram_mb(self.GPU_INDEX)
        self.assertIsNotNone(after_mb, "nvidia-smi unavailable after cycles")

        delta_mb = after_mb - baseline_mb
        self.assertLessEqual(
            delta_mb,
            self.VRAM_TOLERANCE_MB,
            f"VRAM grew by {delta_mb} MiB after 2 destroy() cycles — possible leak. "
            f"Baseline: {baseline_mb} MiB, after: {after_mb} MiB.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
