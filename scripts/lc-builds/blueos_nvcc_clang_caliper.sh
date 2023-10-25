#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -ne 5 ]]; then
  echo
  echo "You must pass 5 arguments to the script (in this order): "
  echo "   1) compiler version number for nvcc"
  echo "   2) CUDA compute architecture (number only, not 'sm_70' for example)"
  echo "   3) compiler version number for clang. "
  echo "   4) path to caliper cmake directory"
  echo "   5) path to adiak cmake directory"
  echo
  echo "For example: "
  echo "    blueos_nvcc_clang_caliper.sh 10.2.89 70 10.0.1 /usr/workspace/wsb/asde/caliper-lassen/share/cmake/caliper /usr/workspace/wsb/asde/adiak-lassen/lib/cmake/adiak"
  exit
fi

COMP_NVCC_VER=$1
COMP_ARCH=$2
COMP_CLANG_VER=$3
CALI_DIR=$4
ADIAK_DIR=$5
shift 5

BUILD_SUFFIX=lc_blueos-nvcc${COMP_NVCC_VER}-${COMP_ARCH}-clang${COMP_CLANG_VER}
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/blueos/nvcc_clang_X.cmake

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.20.2

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-${COMP_CLANG_VER}/bin/clang++ \
  -DBLT_CXX_STD=c++14 \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=On \
  -DCUDA_SEPARABLE_COMPILATION=On \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-${COMP_NVCC_VER} \
  -DCMAKE_CUDA_COMPILER=/usr/tce/packages/cuda/cuda-${COMP_NVCC_VER}/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=${COMP_ARCH} \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  -DRAJA_PERFSUITE_USE_CALIPER=ON \
  -Dcaliper_DIR=${CALI_DIR} \
  -Dadiak_DIR=${ADIAK_DIR} \
  -DRAJA_PERFSUITE_GPU_BLOCKSIZES=128,256,512,1024 \
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
