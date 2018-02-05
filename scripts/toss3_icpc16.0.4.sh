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
## For details about use and distribution, please read raja-perfsuite/LICENSE.
##

rm -rf build_toss3-icpc-16.0.4 2>/dev/null
mkdir build_toss3-icpc-16.0.4 && cd build_toss3-icpc-16.0.4

module load cmake/3.5.2
module load gcc/4.9.3

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/toss3/icpc_16_0_4.cmake \
  -DENABLE_OPENMP=On \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_toss3-icpc-16.0.4 \
  "$@" \
  ..
