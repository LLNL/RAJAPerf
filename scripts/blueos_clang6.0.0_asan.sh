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

rm -rf build_blueos-clang-6.0.0_asan 2>/dev/null
mkdir build_blueos-clang-6.0.0_asan && cd build_blueos-clang-6.0.0_asan

module load cmake/3.7.2
module load cuda/9.2.88
module load clang/6.0.0
module load spectrum-mpi/rolling-release

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -C ${PERFSUITE_DIR}/host-configs/blueos/clang_6_0_0_asan.cmake \
  -DENABLE_OPENMP=On \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DENABLE_EXAMPLES=Off \
  -DENABLE_TESTS=On \
  -DCMAKE_INSTALL_PREFIX=../install_blueos-clang-6.0.0_asan \
  "$@" \
  ${PERFSUITE_DIR}
