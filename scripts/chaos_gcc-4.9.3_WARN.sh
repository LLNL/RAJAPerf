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

rm -rf build_chaos-gcc-4.9.3_debug 2>/dev/null
mkdir build_chaos-gcc-4.9.3_debug && cd build_chaos-gcc-4.9.3_debug

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${PERFSUITE_DIR}/host-configs/chaos/gcc_4_9_3.cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_OPENMP=On \
  -DPERFSUITE_ENABLE_WARNINGS=On \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_chaos-gcc-4.9.3_debug \
  "$@" \
  ${PERFSUITE_DIR}
