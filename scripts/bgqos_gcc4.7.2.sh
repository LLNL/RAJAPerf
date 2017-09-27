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

rm -rf build-bgqos_gcc-4.7.2 2>/dev/null
mkdir build-bgqos_gcc-4.7.2 && cd build-bgqos_gcc-4.7.2
. /usr/local/tools/dotkit/init.sh && use cmake-3.4.3

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${PERFSUITE_DIR}/host-configs/bgqos/gcc_4_7_2.cmake \
  -DENABLE_OPENMP=On \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install-bgqos_gcc-4.7.2 \
  "$@" \
  ${PERFSUITE_DIR}
