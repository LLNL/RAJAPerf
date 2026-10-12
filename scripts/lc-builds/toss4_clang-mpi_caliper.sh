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
  echo "   1) compiler version number"
  echo "   2) path to caliper cmake directory"
  echo "   3) path to adiak cmake directory"
  echo "   4) optional CMake version to load."
  echo
  echo "For example: "
  echo "    toss4_clang-mpi_caliper.sh 14.0.6 /usr/workspace/wsb/asde/caliper-quartz/share/cmake/caliper /usr/workspace/wsb/asde/caliper-quartz/lib/cmake/adiak  [3.27.4]"
  exit 1
fi

COMP_VER=$1
CALI_DIR=$2
ADIAK_DIR=$3

# Detect optional fourth positional argument as a CMake version if it looks like N.M or N.M.P
# Otherwise, treat it as a normal CMake argument.
if [ -n "$4" ] && [[ "$4" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
  CMAKE_VER=$4
  shift 4
else
  CMAKE_VER=$DEFAULT_CMAKE_VER
  shift 3
fi

BUILD_SUFFIX=lc_toss4-clang-mpi-${COMP_VER}
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/toss4/clang_X.cmake

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Using CMake version: ${CMAKE_VER}"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/${CMAKE_VER}

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-${COMP_VER}/bin/clang++ \
  -DBLT_CXX_STD=c++20 \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_MPI=ON \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  -DRAJA_PERFSUITE_USE_CALIPER=ON \
  -Dcaliper_DIR=${CALI_DIR} \
  -Dadiak_DIR=${ADIAK_DIR} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJA Perf Suite"
echo "***********************************************************************"
