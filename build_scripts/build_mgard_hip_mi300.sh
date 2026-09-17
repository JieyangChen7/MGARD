#!/bin/sh

# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
# Script for building MGARD-X
#
# Historical note for multi-GPU MI300 nodes (e.g. odyssey, 4x MI300A): this
# used to require HIP_VISIBLE_DEVICES=0 to avoid an illegal-memory-access
# segfault in the Huffman/norm reduction stages when more than one GPU was
# visible. That was actually an MGARD-X bug, not a ROCm/driver issue:
# DeviceSpecification's and DeviceQueues's per-device query/setup loops in
# DeviceAdapterHip.h called hipSetDevice(d) for every device without
# restoring the originally-active one afterward, leaving the real active
# HIP device out of sync with the library's own bookkeeping on any node
# with more than one GPU. Fixed; HIP_VISIBLE_DEVICES should no longer be
# necessary (verified with repeated runs of the affected tests on odyssey
# with all 4 GPUs visible).

set -e
set -x

module load amd/7.2.0
module load cmake/3.31.1
module load openmpi

######## User Configurations ########
# Source directory
mgard_x_src_dir=.
# Build directory
build_dir=./build-hip-mi300
# Number of processors used for building
num_build_procs=$1
# Installtaion directory
install_dir=./install-hip-mi300

export LD_LIBRARY_PATH=$(pwd)/${install_dir}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/${install_dir}/lib64:$LD_LIBRARY_PATH
export CC=amdclang
export CXX=amdclang++

#build ZSTD
zstd_dir=${build_dir}/zstd
zstd_src_dir=${zstd_dir}/src
zstd_build_dir=${zstd_dir}/build
zstd_install_dir=${install_dir}
if [ ! -d "${zstd_src_dir}" ]; then
  git clone -b v1.5.6 https://github.com/facebook/zstd.git ${zstd_src_dir}
fi
mkdir -p ${zstd_build_dir}
cmake -S ${zstd_src_dir}/build/cmake -B ${zstd_build_dir}\
    -DZSTD_MULTITHREAD_SUPPORT=ON\
    -DCMAKE_INSTALL_LIBDIR=lib\
    -DCMAKE_INSTALL_PREFIX=${zstd_install_dir}
cmake --build ${zstd_build_dir} -j ${num_build_procs}
cmake --install ${zstd_build_dir}


#build Protobuf
protobuf_dir=${build_dir}/protobuf
protobuf_src_dir=${protobuf_dir}/src
protobuf_build_dir=${protobuf_dir}/build
protobuf_install_dir=${install_dir}
if [ ! -d "${protobuf_src_dir}" ]; then
  git clone -b v3.21.12 --recurse-submodules https://github.com/protocolbuffers/protobuf.git ${protobuf_src_dir}
fi
mkdir -p ${protobuf_build_dir}
cmake -S ${protobuf_src_dir}/cmake -B ${protobuf_build_dir}\
    -Dprotobuf_BUILD_SHARED_LIBS=ON\
    -Dprotobuf_BUILD_TESTS=OFF\
    -DCMAKE_INSTALL_PREFIX=${protobuf_install_dir}
cmake --build ${protobuf_build_dir} -j ${num_build_procs}
cmake --install ${protobuf_build_dir}


#build Catch2
catch2_dir=${build_dir}/catch2
catch2_src_dir=${catch2_dir}/src
catch2_build_dir=${catch2_dir}/build
catch2_install_dir=${install_dir}
if [ ! -d "${catch2_src_dir}" ]; then
  git clone -b v3.3.2 https://github.com/catchorg/Catch2.git ${catch2_src_dir}
fi
mkdir -p ${catch2_build_dir}
cmake -S ${catch2_src_dir} -B ${catch2_build_dir}\
    -DBUILD_TESTING=OFF\
    -DCMAKE_INSTALL_PREFIX=${catch2_install_dir}
cmake --build ${catch2_build_dir} -j ${num_build_procs}
cmake --install ${catch2_build_dir} > /dev/null 2>&1

#build MGARD
mgard_x_build_dir=${build_dir}/mgard
mgard_x_install_dir=${install_dir}
mkdir -p ${mgard_x_build_dir}
cmake -S ${mgard_x_src_dir} -B ${mgard_x_build_dir} \
    -DCMAKE_PREFIX_PATH="${zstd_install_dir}/lib/cmake/zstd;${protobuf_install_dir};${catch2_install_dir}"\
    -DMGARD_ENABLE_HIP=ON\
    -DMGARD_ENABLE_OPENMP=OFF\
    -DMGARD_ENABLE_MDR=ON\
    -DCMAKE_HIP_ARCHITECTURES="gfx942"\
    -DCMAKE_BUILD_TYPE=Release\
    -DBUILD_TESTING=ON\
    -DCMAKE_INSTALL_PREFIX=${mgard_x_install_dir}
cmake --build ${mgard_x_build_dir} -j ${num_build_procs}
cmake --install ${mgard_x_build_dir}
