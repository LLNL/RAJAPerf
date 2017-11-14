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

rm -rf build_blueos_nvcc9.0.176_clang-coral-2017.10.13 >/dev/null
mkdir build_blueos_nvcc9.0.176_clang-coral-2017.10.13 && cd build_blueos_nvcc9.0.176_clang-coral-2017.10.13

module load cmake/3.9.2
module load clang/coral-2017.10.13
module load cuda/9.0.176

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${PERFSUITE_DIR}/host-configs/blueos/clang_coral_2017_10_13.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=Off \
  -DOpenMP_CXX_FLAGS="-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-implicit-declare-target" \
  -DRAJA_ENABLE_TARGET_OPENMP=On \
  -DENABLE_TARGET_OPENMP=On \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_blueos_nvcc9.0.176_clang-coral-2017.10.13 \
  "$@" \
  ${PERFSUITE_DIR}
