#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -ne 3 ]]; then
  echo
  echo "You must pass 3 arguments to the script (in this order): "
  echo "   1) compiler version number for nvcc"
  echo "   2) CUDA compute architecture"
  echo "   3) compiler version number for xl. "
  echo
  echo "For example: "
  echo "    blueos_nvcc_xl.sh 11.1.1 sm_70 2021.03.31"
  exit
fi

COMP_NVCC_VER=$1
COMP_ARCH=$2
COMP_XL_VER=$3
shift 3

BUILD_SUFFIX=lc_blueos-nvcc${COMP_NVCC_VER}-${COMP_ARCH}-xl${COMP_XL_VER}
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/blueos/nvcc_xl_X.cmake

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.14.5

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/xl/xl-${COMP_XL_VER}/bin/xlc++_r \
  -DBLT_CXX_STD=c++11 \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=On \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-${COMP_NVCC_VER} \
  -DCMAKE_CUDA_COMPILER=/usr/tce/packages/cuda/cuda-${COMP_NVCC_VER}/bin/nvcc \
  -DCUDA_ARCH=${COMP_ARCH} \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJA Perf Suite"
echo
echo "  Please note that you have to disable CUDA GPU hooks when you run"
echo "  the RAJA Perf Suite; for example,"
echo
echo "    lrun -1 --smpiargs="-disable_gpu_hooks" ./bin/raja-perf.exe"
echo
echo "***********************************************************************"
