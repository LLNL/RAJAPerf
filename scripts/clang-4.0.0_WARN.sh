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

rm -rf build_chaos-clang-4.0.0_debug 2>/dev/null
mkdir build_chaos-clang-4.0.0_debug && cd build_chaos-clang-4.0.0_debug

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${PERFSUITE_DIR}/host-configs/chaos/clang_4_0_0.cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_OPENMP=On \
  -DPERFSUITE_ENABLE_WARNINGS=On \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_chaos-clang-4.0.0_debug \
  "$@" \
  ${PERFSUITE_DIR}
