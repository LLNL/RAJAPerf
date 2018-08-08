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

BUILD_SUFFIX=lc_chaos-clang-6.0.0
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/chaos/clang_6_0_0.cmake

rm -rf build_lc_chaos-clang-6.0.0 2>/dev/null
mkdir build_lc_chaos-clang-6.0.0 && cd build_lc_chaos-clang-6.0.0
. /usr/local/tools/dotkit/init.sh && use cmake-3.4.1

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=../install_lc_chaos-clang-6.0.0 \
  "$@" \
  ..
