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

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_COMPILER "/usr/global/tools/clang/chaos_5_x86_64_ib/clang-cuda-beta-2017-05-30//rawbin/clang++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/global/tools/clang/chaos_5_x86_64_ib/clang-cuda-beta-2017-05-30//rawbin/clang" CACHE PATH "")

set(CMAKE_EXE_LINKER_FLAGS "-rpath /usr/global/tools/clang/chaos_5_x86_64_ib/clang-cuda-beta-2017-05-30//lib:/usr/apps/gnu/4.9.3/lib64:/usr/apps/gnu/4.9.3/lib" CACHE STRING "")

#set(CUDA_COMMON_OPT_FLAGS "--cuda-gpu-arch=sm_37" CACHE STRING "") 

#set(HOST_OPT_FLAGS -Xcompiler -O3)

set(CMAKE_CXX_FLAGS_RELEASE "-O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(BLT_CUDA_ARCH "sm_37" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(BLT_CUDA_ARCH "sm_37" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(BLT_CUDA_ARCH "sm_30" CACHE STRING "")
endif()

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")
set(RAJA_COHERENCE_BLOCK_SIZE 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
