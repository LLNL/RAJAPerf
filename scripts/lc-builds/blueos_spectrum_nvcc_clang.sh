#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -lt 4 ]]; then
  echo
  echo "You must pass 4 arguments to the script (in this order): "
  echo "   1) compiler version number for spectrum mpi"
  echo "   2) compiler version number for nvcc"
  echo "   3) CUDA compute architecture"
  echo "   4) compiler version number for clang. "
  echo
  echo "For example: "
  echo "    blueos_nvcc_clang.sh rolling-release 10.2.89 sm_70 10.0.1"
  exit
fi

COMP_MPI_VER=$1
COMP_NVCC_VER=$2
COMP_ARCH=$3
COMP_CLANG_VER=$4
shift 4

BUILD_SUFFIX=lc_blueos-spectrum${COMP_MPI_VER}-nvcc${COMP_NVCC_VER}-${COMP_ARCH}-clang${COMP_CLANG_VER}
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/blueos/nvcc_clang_X.cmake

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
  -DMPI_CXX_COMPILER=/usr/tce/packages/spectrum-mpi/spectrum-mpi-${COMP_MPI_VER}-clang-${COMP_CLANG_VER}/bin/mpiclang++ \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-${COMP_CLANG_VER}/bin/clang++ \
  -DBLT_CXX_STD=c++14 \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_MPI=On \
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
echo "  Please note that you have to run with mpi when you run"
echo "  the RAJA Perf Suite; for example,"
echo
echo "    lrun -n4 ./bin/raja-perf.exe"
echo
echo "***********************************************************************"
