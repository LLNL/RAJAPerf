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

rm -rf build_chaos-gcc-4.9.3_debug 2>/dev/null
mkdir build_chaos-gcc-4.9.3_debug && cd build_chaos-gcc-4.9.3_debug
. /usr/local/tools/dotkit/init.sh && use cmake-3.4.1

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -C ${PERFSUITE_DIR}/host-configs/chaos/gcc_4_9_3.cmake \
  -DENABLE_OPENMP=On \
  -DPERFSUITE_ENABLE_WARNINGS=On \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_chaos-gcc-4.9.3_debug \
  "$@" \
  ${PERFSUITE_DIR}
