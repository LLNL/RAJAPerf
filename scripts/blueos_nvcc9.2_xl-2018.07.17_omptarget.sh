#!/bin/bash

##
## Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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

rm -rf build_blueos_nvcc9.2_xl-2018.07.17_omptarget >/dev/null
mkdir build_blueos_nvcc9.2_xl-2018.07.17_omptarget && cd build_blueos_nvcc9.2_xl-2018.07.17_omptarget

module load cmake/3.7.2
module load cuda/9.2.88
module load xl/beta-2018.07.17
module load spectrum-mpi/rolling-release

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release\
  -C ${PERFSUITE_DIR}/host-configs/blueos/nvcc_xl_2018_07_17_omptarget.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_TARGET_OPENMP=On \
  -DOpenMP_CXX_FLAGS="-qoffload -qsmp=omp -qnoeh -qnoinline -qalias=noansi" \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-9.2.88 \
  -DENABLE_ALL_WARNINGS=Off \
  -DENABLE_EXAMPLES=Off \
  -DENABLE_TESTS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_blueos_nvcc9.2_xl-2018.07.17_omptarget \
  "$@" \
  ${PERFSUITE_DIR}
