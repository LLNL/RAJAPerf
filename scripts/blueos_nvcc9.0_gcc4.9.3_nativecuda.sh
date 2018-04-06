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

rm -rf build_blueos_nvcc9.0_gcc4.9.3_native >/dev/null
mkdir build_blueos_nvcc9.0_gcc4.9.3_native && cd build_blueos_nvcc9.0_gcc4.9.3_native

module load cmake/3.8.2

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/blueos/nvcc_gcc_4_9_3_native.cmake \
  -DENABLE_OPENMP=Off \
  -DENABLE_EXAMPLES=Off \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_blueos_nvcc9.0_gcc4.9.3_native \
  "$@" \
  ..
