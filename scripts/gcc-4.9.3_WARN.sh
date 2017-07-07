#!/bin/bash

##
## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## For additional details and restrictions, please see RAJA/LICENSE.txt
##

rm -rf build-gcc-4.9.3-debug_WARN 2>/dev/null
mkdir build-gcc-4.9.3-debug_WARN && cd build-gcc-4.9.3-debug_WARN

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

RAJA_DIR=${PERFSUITE_DIR}/tpl/RAJA

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/gcc_4_9_3.cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPERFSUITE_ENABLE_WARNINGS=On \
  -DENABLE_OPENMP=On \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install-gcc-4.9.3 \
  "$@" \
  ${PERFSUITE_DIR}
