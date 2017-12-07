#!/bin/bash

##
## Copyright (c) 2017, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-738930
##
## All rights reserved.
## 
## This file is part of the RAJA Performance Suite.
##
## For details about use and distribution, please read raja-perfsuite/LICENSE.
##

rm -rf build_blueos_nvcc9.0_xl-beta-2017.11.28 >/dev/null
mkdir build_blueos_nvcc9.0_xl-beta-2017.11.28 && cd build_blueos_nvcc9.0_xl-beta-2017.11.28

module load cmake/3.7.2

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${PERFSUITE_DIR}/host-configs/blueos/nvcc_xl-beta-2017.11.28.cmake \
  -DENABLE_MODULES=Off \
  -DENABLE_TESTS=Off \
  -DENABLE_OPENMP=On \
  -DENABLE_TARGET_OPENMP=On \
  -DOpenMP_CXX_FLAGS="-qsmp=omp -qoffload" \
  -DENABLE_CUDA=Off \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tcetmp/packages/cuda-9.0.176 \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_blueos_nvcc9.0_xl-beta-2017.11.28 \
  "$@" \
  ${PERFSUITE_DIR}
