#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [ "$1" == "" ]; then
  echo
  echo "You must pass a compiler version number to script. For example,"
  echo "    ubuntu_clang.sh 10"
  exit
fi

COMP_VER=$1
shift 1

BUILD_SUFFIX=ubuntu-clang-${COMP_VER}
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/ubuntu-builds/clang_X.cmake

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-${COMP_VER} \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-${COMP_VER} \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  .. 

echo
echo "***********************************************************************"
echo "cd into directory ${BUILD_SUFFIX} and run make to build RAJA Perf Suite"
echo "***********************************************************************"
