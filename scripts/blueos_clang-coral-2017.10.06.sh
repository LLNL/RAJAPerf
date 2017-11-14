
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


rm -rf build_blueos_clang-coral-2017.10.06 >/dev/null
mkdir build_blueos_clang-coral-2017.10.06 && cd build_blueos_clang-coral-2017.10.06

module load cmake/3.7.2
module load clang/coral-2017.10.06
module load cuda/9.0.176 

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${PERFSUITE_DIR}/host-configs/blueos/clang_coral_2017_10_06.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=Off \
  -DOpenMP_CXX_FLAGS="-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-implicit-declare-target" \
  -DRAJA_ENABLE_TARGET_OPENMP=On \
  -DRAJA_ENABLE_MODULES=Off \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tcetmp/packages/cuda-9.0.176 \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DRAJA_BUILD_WITH_BLT=On \
  -DCMAKE_INSTALL_PREFIX=../install_blueos_clang-coral-2017.10.06 \
  "$@" \
  ${PERFSUITE_DIR}
