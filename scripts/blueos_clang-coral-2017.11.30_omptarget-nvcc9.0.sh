
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


rm -rf build_blueos_clang-coral-2017.11.30 >/dev/null
mkdir build_blueos_clang-coral-2017.11.30 && cd build_blueos_clang-coral-2017.11.30


export CUDA_DIR=$CUDA_HOME

PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${PERFSUITE_DIR}/host-configs/blueos/clang_coral_2017_11_30.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_TARGET_OPENMP=On \
  -DENABLE_TESTS=Off \
  -DENABLE_CUDA=Off \
  -DOpenMP_CXX_FLAGS=" -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-implicit-declare-target -fopenmp-implicit-map-lambdas" \
  -DRAJA_ENABLE_MODULES=Off \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_blueos_clang-coral-2017.11.30 \
  "$@" \
  ${PERFSUITE_DIR}
