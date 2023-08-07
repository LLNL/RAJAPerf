#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -ne 3 ]]; then
  echo
  echo "You must pass 3 arguments to the script (in this order): "
  echo "   1) compiler version number"
  echo "   2) path to caliper cmake directory"
  echo "   3) path to adiak cmake directory"
  echo
  echo "For example: "
  echo "    toss4_gcc_caliper.sh 10.3.1 /usr/workspace/wsb/asde/caliper-quartz/share/cmake/caliper /usr/workspace/wsb/asde/caliper-quartz/lib/cmake/adiak"
  exit
fi

COMP_VER=$1
CALI_DIR=$2
ADIAK_DIR=$3
shift 3

BUILD_SUFFIX=lc_toss4-gcc-${COMP_VER}
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/toss4/gcc_X.cmake

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.21.1

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/gcc/gcc-${COMP_VER}/bin/g++ \
  -DBLT_CXX_STD=c++14 \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  -DRAJA_PERFSUITE_USE_CALIPER=ON \
  -Dcaliper_DIR=${CALI_DIR} \
  -Dadiak_DIR=${ADIAK_DIR} \
  -DCMAKE_C_FLAGS="-g -O0" \
  -DCMAKE_CXX_FLAGS="-g -O0" \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJA Perf Suite"
echo "***********************************************************************"
