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

rm -rf build_chaos-icpc-16.0.258 2>/dev/null
mkdir build_chaos-icpc-16.0.258 && cd build_chaos-icpc-16.0.258
. /usr/local/tools/dotkit/init.sh && use cmake-3.4.1 && use gcc-4.9.3p

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${PERFSUITE_DIR}/host-configs/chaos/icpc_16_0_258.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_TBB=Off \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_chaos-icpc-16.0.258 \
  "$@" \
  ${PERFSUITE_DIR}
