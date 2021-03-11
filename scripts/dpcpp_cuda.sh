#!/usr/bin/env bash

##
## Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
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


BUILD_SUFFIX=dpcpp_cuda
: ${BUILD_TYPE:=RelWithDebInfo}
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/alcf-builds/dpcpp.cuda.cmake

rm -rf build_${BUILD_SUFFIX}_${USER} >/dev/null
mkdir build_${BUILD_SUFFIX}_${USER} && cd build_${BUILD_SUFFIX}_${USER}

cmake \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=Off \
  -DENABLE_CUDA=Off \
  -DENABLE_TARGET_OPENMP=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DENABLE_SYCL=On \
  -DCMAKE_CXX_STANDARD=14 \
  -DCMAKE_LINKER=clang++ \
  "$@" \
  ..

make -j 18
