##
## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## For release details and restrictions, please see RAJA/LICENSE.
##

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_COMPILER "/usr/tcetmp/packages/clang/clang-coral-2017.06.29/bin/clang++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tcetmp/packages/clang/clang-coral-2017.06.29/bin/clang" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g" CACHE STRING "")

set(CUDA_COMMON_FLAGS -restrict; -arch sm_60; -std c++11; --expt-extended-lambda)

set(HOST_OPT_FLAGS -Xcompiler -O3 -Xcompiler -fopenmp)

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(BLT_CUDA_FLAGS -O3; ${CUDA_COMMON_FLAGS}; -ccbin; ${CMAKE_CXX_COMPILER} ; ${HOST_OPT_FLAGS} CACHE LIST "")
  set(RAJA_NVCC_FLAGS -O3; ${CUDA_COMMON_FLAGS}; -ccbin; ${CMAKE_CXX_COMPILER} CACHE LIST "")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(BLT_CUDA_FLAGS -g; -G; -O3; ${CUDA_COMMON_FLAGS}; -ccbin; ${CMAKE_CXX_COMPILER} ; ${HOST_OPT_FLAGS} CACHE LIST "")
  set(RAJA_NVCC_FLAGS -g; -G; -O3; ${CUDA_COMMON_FLAGS}; -ccbin; ${CMAKE_CXX_COMPILER} ; CACHE LIST "")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(BLT_CUDA_FLAGS -g; -G; -O0; ${CUDA_COMMON_FLAGS}; -ccbin; ${CMAKE_CXX_COMPILER} ; -Xcompiler -fopenmp CACHE LIST "")
  set(RAJA_NVCC_FLAGS -g; -G; -O0; ${CUDA_COMMON_FLAGS}; -ccbin ; ${CMAKE_CXX_COMPILER} CACHE LIST "")
endif()

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")
set(RAJA_COHERENCE_BLOCK_SIZE 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
