#!/bin/bash

##
## Copyright (c) 2017, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-xxxxxx
##
## All rights reserved.
##
## This file is part of the RAJA Performance Suite.
##
## For more information, see the file LICENSE in the top-level directory.
##

rm -rf build_blueos-nvcc8.0_clang4.0.0 2>/dev/null
mkdir build_blueos-nvcc8.0_clang4.0.0 && cd build_blueos-nvcc8.0_clang4.0.0

module load cmake/3.7.2
module load clang/4.0.0 
module load cuda/8.0

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=On \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_blueos-nvcc8.0_clang4.0.0  \
  "$@" \
  ${PERFSUITE_DIR}
