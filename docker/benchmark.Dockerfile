# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Benchmark/eval image for nvidia-foundationpose-runtime.
# Base has CUDA 13 + TensorRT 10.13 (Blackwell sm_120 capable). Adds the headless
# GL/EGL system libs that bop_toolkit's vispy renderer (PyOpenGL+EGL) needs, plus
# the Python benchmark + BOP-eval deps in an isolated venv (/opt/bench).
FROM nvcr.io/nvidia/pytorch:26.05-py3
ENV DEBIAN_FRONTEND=noninteractive
# Isolated venv, pinned to the exact versions this runtime was validated with
# (numpy stays <2 as bop_toolkit requires). bop_toolkit itself is installed
# editable at run time from the mounted dataset tree (validated commit 0d62c3e7).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglvnd0 libgl1 libglx0 libegl1 libgles2 libopengl0 \
      libfontconfig1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && python -m venv /opt/bench \
 && /opt/bench/bin/pip install --no-cache-dir -q --upgrade pip \
 && /opt/bench/bin/pip install --no-cache-dir -q \
      numpy==1.26.4 opencv-python-headless==4.11.0.86 scipy==1.17.1 \
      imageio==2.37.3 pypng==0.20220715.0 pillow==12.2.0 vispy==0.16.2 \
      PyOpenGL==3.1.10 pytz==2026.2 scikit-image==0.26.0 \
 && chmod -R 0777 /opt/bench   # allow runtime --user to `pip install -e bop_toolkit`
