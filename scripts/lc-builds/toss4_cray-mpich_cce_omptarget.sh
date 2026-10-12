#!/usr/bin/env bash

###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA Performance Suite.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

# Default CMake version if not provided
DEFAULT_CMAKE_VER=3.24.2

if [[ $# -lt 3 ]]; then
  echo
  echo "You must pass 3 or more arguments to the script (in this order): "
  echo "   1) cray-mpich version number"
  echo "   2) compiler version number"
  echo "   3) HIP compute architecture"
  echo "   4) optional CMake version to load."
  echo
  echo "For example: "
  echo "    toss4_cray-mpich_cce_omptarget.sh 9.0.1 20.0.0-magic gfx942 [3.27.4]"
  exit 1
fi

MPI_VER=$1
COMP_VER=$2
HIP_ARCH=$3

# Detect optional fourth positional argument as a CMake version if it looks like N.M or N.M.P
# Otherwise, treat it as a normal CMake argument.
if [ -n "$4" ] && [[ "$4" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
  CMAKE_VER=$4
  shift 4
else
  CMAKE_VER=$DEFAULT_CMAKE_VER
  shift 3
fi

HOSTCONFIG="cce_omptarget_X"

BUILD_SUFFIX=lc_toss4-cray-mpich-${MPI_VER}-cce-${COMP_VER}-${HIP_ARCH}-omptarget

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Using CMake version: ${CMAKE_VER}"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}


module load cmake/${CMAKE_VER}

if [[ "${COMP_VER}" == *-magic ]]; then
  MPI_PATH="/usr/tce/packages/cray-mpich/cray-mpich-${MPI_VER}-cce-${COMP_VER}"
  COMP_PATH="/usr/tce/packages/cce/cce-${COMP_VER}"
else
  MPI_PATH="/usr/tce/packages/cray-mpich-tce/cray-mpich-${MPI_VER}-cce-${COMP_VER}"
  COMP_PATH="/usr/tce/packages/cce-tce/cce-${COMP_VER}"
fi

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI_C_COMPILER="${MPI_PATH}/bin/mpicc" \
  -DMPI_CXX_COMPILER="${MPI_PATH}/bin/mpicxx" \
  -DHIP_ARCH=${HIP_ARCH} \
  -DCMAKE_C_COMPILER="${COMP_PATH}/bin/craycc" \
  -DCMAKE_CXX_COMPILER="${COMP_PATH}/bin/crayCC" \
  -DBLT_CXX_STD=c++20 \
  -DENABLE_CLANGFORMAT=Off \
  -C "../host-configs/lc-builds/toss4/${HOSTCONFIG}.cmake" \
  -DENABLE_MPI=ON \
  -DENABLE_HIP=OFF \
  -DENABLE_OPENMP=ON \
  -DRAJA_ENABLE_TARGET_OPENMP=ON \
  -DENABLE_CUDA=OFF \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJA"
echo
echo "***********************************************************************"
