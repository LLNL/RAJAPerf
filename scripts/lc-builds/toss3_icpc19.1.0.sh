#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#################################################################################

BUILD_SUFFIX=lc_toss3-icpc-19.1.0
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/toss3/icpc_X_gcc8headers.cmake

rm -rf build_${BUILD_SUFFIX}_$1 2>/dev/null
mkdir build_${BUILD_SUFFIX}_$1 && cd build_${BUILD_SUFFIX}_$1

module load cmake/3.14.5

if [ "$1" == "seq" ]; then
    argS="On"
    argV="Off"
    RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/toss3/clang_X.cmake
else
    argS="On"
    argV="On"
    RAJA_HOSTCONFIG=../tpl/RAJAvec/host-configs/lc-builds/toss3/clang_X.cmake
fi

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/intel/intel-19.1.0/bin/icpc \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=On \
  -DENABLE_RAJA_VECTORIZATION=$argV \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
