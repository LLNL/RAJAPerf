#!/bin/bash

##
## Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
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

rm -rf build_blueos_nvcc9.1_clang-coral-2018.02.09 >/dev/null
mkdir build_blueos_nvcc9.1_clang-coral-2018.02.09 && cd build_blueos_nvcc9.1_clang-coral-2018.02.09

module load cmake/3.7.2

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/blueos/nvcc_clang_coral_2018_02_09.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=On \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-9.2.64 \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_blueos_nvcc9.1_clang-coral-2018.02.09 \
  "$@" \
  ..
