#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#################################################################################

BUILD_SUFFIX=lc_toss3-icpc-19.1.0

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX}_$1 && cd build_${BUILD_SUFFIX}_$1

module load cmake/3.14.5

if [ "$1" == "orig" ]; then
    argO="On"
    argV="Off"
    argD="Off"
    RAJA_HOSTCONFIG=../tpl/RAJAorig/host-configs/lc-builds/toss3/clang_X.cmake
elif [ "$1" == "origExt" ]; then
    argO="Off"
    argV="On"
    argD="Off"
    RAJA_HOSTCONFIG=../tpl/RAJAorig/host-configs/lc-builds/toss3/clang_X.cmake
else
    argO="Off"
    argV="Off"
    argD="On"
    RAJA_HOSTCONFIG=../tpl/RAJAdev/host-configs/lc-builds/toss3/clang_X.cmake
fi

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/intel/intel-19.1.0/bin/icpc \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=On \
  -DENABLE_RAJA_SEQUENTIAL=$argO -DENABLE_RAJA_SEQUENTIAL_ARGS=$argV -DENABLE_RAJA_SEQUENTIAL_ARGS_DEV=$argD \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX}_$1 \
  "$@" \
  ..
