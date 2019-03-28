#!/usr/bin/env bash

##
## Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-738930
##
## All rights reserved.
##
## This file is part of the RAJA Performance Suite.
##
## For details about use and distribution, please read RAJAPerf/LICENSE.
##

BUILD_SUFFIX=lc_toss3-clangcuda-6.0.0_nvcc-8.0
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/toss3/clangcuda_6_0_0_nvcc_8_0.cmake

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.9.2

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=Off \
  -DENABLE_CLANG_CUDA=On \
  -DBLT_CLANG_CUDA_ARCH=sm_35 \
  -DENABLE_CUDA=On \
  -DCUDA_ARCH=sm_35 \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-8.0 \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
