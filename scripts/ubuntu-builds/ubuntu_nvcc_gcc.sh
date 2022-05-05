#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -lt 3 ]]; then
  echo
  echo "You must pass 3 arguments to the script (in this order): "
  echo "   1) compiler version number for nvcc"
  echo "   2) CUDA compute architecture"
  echo "   3) compiler version number for gcc. "
  echo
  echo "For example: "
  echo "    ubuntu_nvcc_gcc.sh 10.1.243 sm_61 8"
  exit
fi

COMP_NVCC_VER=$1
COMP_ARCH=$2
COMP_GCC_VER=$3
shift 3

BUILD_SUFFIX=ubuntu-nvcc-${COMP_NVCC_VER}-${COMP_ARCH}-gcc-${COMP_GCC_VER}
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/ubuntu-builds/nvcc_gcc_X.cmake

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-${COMP_GCC_VER} \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-${COMP_GCC_VER} \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCUDA_ARCH=${COMP_ARCH} \
  -DBLT_CXX_STD=c++14 \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJA Perf Suite"
echo "***********************************************************************"
