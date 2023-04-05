#!/bin/bash
# build_docker.sh

BASE_IMAGE=arm64v8/ubuntu:20.04
DOCKER_TAG=knn-lib-arm64v8-builds:pynq27
CMAKE_VERSION=cmake-3.25.2-linux-aarch64
PYTHON_VERSION=3.8.2

echo "CMake version: ${CMAKE_VERSION}"
echo "Python version: ${PYTHON_VERSION}"

# Build PyTorch using docker
docker build --build-arg BASE_IMAGE="${BASE_IMAGE}" \
             --build-arg CMAKE_VERSION="${CMAKE_VERSION}" \
             --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
             --cpuset-cpus="0-7" \
             -t "${DOCKER_TAG}" \
             -f Dockerfile .

# Copy the wheel
docker run -v "${PWD}:/opt/mount" --rm "${DOCKER_TAG}" \
  bash -c "cp /*.whl /opt/mount && \
    chown $(id -u):$(id -g) /opt/mount/*.whl"

