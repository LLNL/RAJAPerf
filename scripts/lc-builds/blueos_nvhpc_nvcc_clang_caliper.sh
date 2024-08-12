#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

module load cmake/3.20.2
module load nvhpc/24.1-cuda-11.2.0

BUILD_SUFFIX=lc_blueos-nvhpc-nvcc-clang-caliper
rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_FLAGS_RELWITHDEBINFO="-O3" \
    -DBLT_CXX_STD=c++14 \
    -DRAJA_PERFSUITE_USE_CALIPER=On \
    -DENABLE_CUDA=on \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-ibm-16.0.6-cuda-11.2.0-gcc-8.3.1/bin/clang++ \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/nvhpc/nvhpc-24.1-cuda-11.2.0/Linux_ppc64le/24.1/cuda/11.2/ \
    -DCMAKE_CUDA_COMPILER=/usr/tce/packages/nvhpc/nvhpc-24.1-cuda-11.2.0/Linux_ppc64le/24.1/cuda/11.2/bin/nvcc \
    -DRAJA_PERFSUITE_GPU_BLOCKSIZES=128,256,512,1024 \
    -DRAJA_PERFSUITE_TUNING_CUDA_ARCH=700 \
    -Dcaliper_DIR=/usr/workspace/wsb/asde/caliper-lassen/share/cmake/caliper \
    -Dadiak_DIR=/usr/workspace/wsb/asde/caliper-lassen/lib/cmake/adiak \
    ..
#-DRAJA_PERFSUITE_ENABLE_TESTS=OFF \
