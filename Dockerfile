# Academic Software License: Copyright © 2025 Goodarz Mehr.

# SimBEV Docker Configuration Script
#
# This script performs all the necessary steps for creating a SimBEV Docker
# image.
#
# The base Docker image is Ubuntu 22.04 with CUDA 13.0 and Vulkan SDK
# 1.3.204.1. If you want to use a different base image, you may have to modify
# "ubuntu2204/x86_64" when fetching keys, based on your Ubuntu release and
# system architecture.

# Build Arguments (Case Sensitive):
#
# USER:          username inside each container, set to "sb" by default.
# CARLA_VERSION: installed version of CARLA, set to "0.9.16" by default.

# Installation:
#
# 1. Install Docker on your system (https://docs.docker.com/engine/install/).
# 2. Install the Nvidia Container Toolkit
# (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide).
# It exposes your Nvidia graphics card to Docker containers.
# 3. In the Dockerfile directory, run
#
# docker build --no-cache --rm --build-arg ARG -t simbev:develop .

# Usage:
#
# Launch a container by running
#
# docker run --privileged --gpus all --network=host -e DISPLAY=$DISPLAY
# -v [path/to/CARLA]:/home/carla
# -v [path/to/SimBEV]:/home/simbev
# -v [path/to/dataset]:/dataset
# --shm-size 32g -it simbev:develop /bin/bash
#
# Use "nvidia-smi" to ensure your graphics card is visible inside the
# container.

FROM nvidia/cuda:13.0.1-devel-ubuntu22.04

# Define build arguments and environment variables.

ARG USER=sb
ARG CARLA_VERSION=0.9.16

ENV USER=${USER}
ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive
ENV CARLA_VERSION=$CARLA_VERSION
ENV CARLA_ROOT=/home/carla

# Add new user and install prerequisite packages.

WORKDIR /home

RUN useradd -m ${USER}

RUN set -xue && apt-key del 7fa2af80 \
&& apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
&& apt-get update \
&& apt-get install -y build-essential cmake debhelper git wget xdg-user-dirs xserver-xorg libvulkan1 libsdl2-2.0-0 \
libsm6 libgl1-mesa-glx libomp5 pip unzip libjpeg8 libtiff5 software-properties-common nano fontconfig g++ gcc gdb \
libglib2.0-0 libgtk2.0-dev libnvidia-gl-580 libnvidia-common-580 libvulkan-dev vulkan-tools python-is-python3 \
mesa-utils

RUN pip install --no-cache-dir ninja numpy matplotlib opencv-python open3d scikit-image flow_vis pyquaternion \
networkx psutil tqdm pynput pyspng

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130

USER ${USER}