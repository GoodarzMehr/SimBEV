# MIT License
#
# Copyright (C) 2023 Goodarz Mehr
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

# SimBEV Docker Configuration Script
#
# Performs all the necessary tasks for the generation of a generic SimBEV
# Docker image.
#
# The base Docker image is Ubuntu 20.04 with CUDA 11.3 and Vulkan SDK
# 1.3.204.1. If you want to use a different base image, you may need to modify
# "ubuntu2004/x86_64" when fetching keys, according to your Ubuntu release and
# system architecture.

# Build Arguments (Case Sensitive):
#
# USER:            default username inside each container, set to "sb" by
#                  default.
# CARLA_VERSION:   desired version of CARLA, set to "0.9.14" by default.

# Installation:
#
# 1. Install Docker on your system (https://docs.docker.com/engine/install/).
#
# 2. Install the Nvidia Container Toolkit
# (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide).
# It exposes your Nvidia graphics card to Docker containers.
#
# 3. In the Dockerfile directory, run
#
# docker build --no-cache --rm --build-arg ARG -t simbev:develop .

# Usage:
#
# Launch a container by running
#
# docker run --privileged --gpus all --network=host -e DISPLAY=$DISPLAY
# -v [path/to/SimBEV]/simbev:/home/simbev
# -v [path/to/dataset]/data:/dataset
# --shm-size 32g -it simbev:develop /bin/bash
#
# Use "nvidia-smi" and "nvcc --version" to ensure your graphics card and CUDA
# are both visible inside the container.

FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Define build arguments and environment variables.

ARG USER=sb
ARG CARLA_VERSION=0.9.14

ENV USER=${USER}
ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive
ENV CARLA_VERSION=$CARLA_VERSION
ENV CARLA_ROOT=/home/carla
ENV PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla/dist/carla-${CARLA_VERSION}-py3.7-linux-x86_64.egg

# Add new user and install prerequisite packages.

WORKDIR /home

RUN useradd -m ${USER}

RUN set -xue && apt-key del 7fa2af80 \
&& apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
&& apt-get update \
&& apt-get install -y build-essential cmake debhelper git wget xdg-user-dirs xserver-xorg libvulkan1 libsdl2-2.0-0 \
libsm6 libgl1-mesa-glx libomp5 pip unzip libjpeg8 libtiff5 software-properties-common nano fontconfig g++ gcc gdb \
libglib2.0-0 libgtk2.0-dev libnvidia-gl-470 libnvidia-common-470 libvulkan-dev vulkan-utils python-is-python3

RUN pip install numpy matplotlib opencv-python open3d

COPY --chown=${USER}:${USER} carla ${CARLA_ROOT}

USER ${USER}
