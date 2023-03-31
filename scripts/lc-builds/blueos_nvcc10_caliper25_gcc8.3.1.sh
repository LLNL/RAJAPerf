#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#################################################################################

BUILD_SUFFIX=lc_blueos-nvcc10-caliper-gcc8.3.1
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/blueos/nvcc_gcc_X.cmake

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.14.5
module load caliper-2.5.0-gcc-8.3.1-cu3vy3k

CALIPER_PREFIX=/usr/WS2/holger/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/caliper-2.5.0-cu3vy3kjwjerpdm6xis2kauhz4s6wto2/

ADIAK_PREFIX=/usr/WS2/holger/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/adiak-0.2.1-hsv444o7ofb6s2znkvvnh6hcmr774g73/

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/gcc/gcc-8.3.1/bin/g++ \
  -C ${RAJA_HOSTCONFIG} \
  -DCMAKE_PREFIX_PATH="${CALIPER_PREFIX}/share/cmake/caliper;${ADIAK_PREFIX}/lib/cmake/adiak" \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=On \
  -DCMAKE_CUDA_FLAGS="-Xcompiler -mno-float128" \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-10.2.89 \
  -DCMAKE_CUDA_COMPILER=/usr/tce/packages/cuda/cuda-10.1.243/bin/nvcc \
  -DCUDA_ARCH=sm_70 \
  -DENABLE_CALIPER=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
  -DCMAKE_VERBOSE_MAKEFILE=On \
  "$@" \
  ..
