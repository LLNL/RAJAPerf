#!/usr/bin/env bash

###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA Performance Suite.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -lt 2 ]]; then
  echo
  echo "You must pass 2 or more arguments to the script (in this order): "
  echo "   1) compiler version number"
  echo "   2) HIP compute architecture"
  echo "   3...) optional arguments to cmake"
  echo
  echo "For example: "
  echo "    toss4_amdclang.sh 6.4.1 gfx1201"
  exit
fi

COMP_VER=$1
COMP_ARCH=$2
shift 2

HOSTCONFIG="clang_X"

BUILD_SUFFIX=lc_toss4-amdclang-${COMP_VER}-${COMP_ARCH}
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/ubuntu-builds/${HOSTCONFIG}.cmake

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo
echo "To get cmake to work you may have to configure with"
echo "   -DHIP_PLATFORM=amd"
echo
echo "To use fp64 HW atomics you must configure with these options when using gfx90a and hip >= 5.2"
echo "   -DCMAKE_CXX_FLAGS=\"-munsafe-fp-atomics\""
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}


# unload rocm to avoid configuration problems where the loaded rocm and COMP_VER
# are inconsistent causing the rocprim from the module to be used unexpectedly
# module unload rocm

ROCM_PATH="/opt/rocm-${COMP_VER}"

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCM_ROOT_DIR="${ROCM_PATH}" \
  -DHIP_ROOT_DIR="${ROCM_PATH}/hip" \
  -DHIP_PATH=${ROCM_PATH}/llvm/bin \
  -DCMAKE_C_COMPILER=${ROCM_PATH}/llvm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES="${COMP_ARCH}" \
  -DGPU_TARGETS="${COMP_ARCH}" \
  -DAMDGPU_TARGETS="${COMP_ARCH}" \
  -DBLT_CXX_STD=c++20 \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_HIP=ON \
  -DENABLE_OPENMP=ON \
  -DENABLE_CUDA=OFF \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJAPerf"
echo
echo "***********************************************************************"
