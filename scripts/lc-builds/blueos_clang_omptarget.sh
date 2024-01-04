#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -lt 1 ]]; then
  echo
  echo "You must pass a compiler version number to script. For example,"
  echo "    blueos_clang_omptarget.sh 10.0.1-gcc-8.3.1"
  echo "  - or -"
  echo "    blueos_clang_omptarget.sh ibm-10.0.1-gcc-8.3.1"
  exit
fi

COMP_VER=$1
shift 1

BUILD_SUFFIX=lc_blueos-clang-${COMP_VER}_omptarget
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/blueos/clang_X.cmake

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.20.2

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-${COMP_VER}/bin/clang++ \
  -DBLT_CXX_STD=c++14 \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=Off \
  -DRAJA_ENABLE_TARGET_OPENMP=On \
  -DBLT_OPENMP_COMPILE_FLAGS="-fopenmp;-fopenmp-targets=nvptx64-nvidia-cuda" \
  -DBLT_OPENMP_LINK_FLAGS="-fopenmp;-fopenmp-targets=nvptx64-nvidia-cuda" \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJA Perf Suite"
echo "***********************************************************************"

