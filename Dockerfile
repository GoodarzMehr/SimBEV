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

# Installation:
#
# 1. In the Dockerfile directory, run
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

FROM simbev-base:develop

COPY --chown=${USER}:${USER} carla ${CARLA_ROOT}

USER ${USER}
