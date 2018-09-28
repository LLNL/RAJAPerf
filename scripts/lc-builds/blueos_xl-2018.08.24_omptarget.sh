#!/bin/bash

##
## Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-738930
##
## All rights reserved.
##
## This file is part of the RAJA Performance Suite.
##
## For details about use and distribution, please read RAJAPerf/LICENSE.
##

BUILD_SUFFIX=lc_blueos-xl_2018.08.24_omptarget
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/blueos/xl_2018_08_24.cmake

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.9.2

cmake \
  -DCMAKE_BUILD_TYPE=Release\
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=On \
  -DENABLE_TARGET_OPENMP=On \
  -DENABLE_EXAMPLES=Off \
  -DENABLE_TESTS=Off \
  -DOpenMP_CXX_FLAGS="-std=c++11 -qoffload -qsmp=omp -qnoeh -qnoinline -qalias=noansi" \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
