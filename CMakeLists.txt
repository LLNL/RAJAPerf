###############################################################################
# Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

# C is required for googletest to find Threads
project(RAJAPerfSuite LANGUAGES CXX C)

cmake_minimum_required(VERSION 3.14.5)

option(ENABLE_RAJA_SEQUENTIAL "Run sequential variants of RAJA kernels. Disable
this, and all other variants, to run _only_ raw C loops." On)
option(ENABLE_KOKKOS "Include Kokkos implementations of the kernels in the RAJA Perfsuite" Off)

#
# Note: the BLT build system is inheritted by RAJA and is initialized by RAJA
#

if (PERFSUITE_ENABLE_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()

if(ENABLE_KOKKOS)
  set(CMAKE_CXX_STANDARD 17)
  set(BLT_CXX_STD c++17)
elseif (ENABLE_STDPAR)
  set(CMAKE_CXX_STANDARD 20)
  set(BLT_CXX_STD c++14)
else()
  set(CMAKE_CXX_STANDARD 14)
  set(BLT_CXX_STD c++14)
endif()

include(blt/SetupBLT.cmake)

#
# Define RAJA PERFSUITE settings...
#

cmake_dependent_option(RAJA_PERFSUITE_ENABLE_TESTS "Enable RAJA Perf Suite Tests" On "ENABLE_TESTS" Off)

if (ENABLE_TESTS)

  set(RAJA_ENABLE_TESTS Off CACHE BOOL "")
  set(CAMP_ENABLE_TESTS Off CACHE BOOL "")

endif()

cmake_dependent_option(RAJA_PERFSUITE_ENABLE_MPI "Build with MPI" On "ENABLE_MPI" Off)

cmake_dependent_option(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN "Build OpenMP scan variants" Off "ENABLE_OPENMP" Off)

#
# Define RAJA settings...
#

set(RAJA_ENABLE_TESTS Off CACHE BOOL "")
set(RAJA_ENABLE_EXAMPLES Off CACHE BOOL "")
set(RAJA_ENABLE_EXERCISES Off CACHE BOOL "")
set(ENABLE_DOCUMENTATION Off CACHE BOOL "")

set(ENABLE_TBB Off CACHE BOOL "")

set(RAJA_USE_CHRONO On CACHE BOOL "")

set(RAJA_PERFSUITE_GPU_BLOCKSIZES "" CACHE STRING "Comma separated list of GPU block sizes, ex '256,1024'")

set(RAJA_RANGE_ALIGN 4)
set(RAJA_RANGE_MIN_LENGTH 32)
set(RAJA_DATA_ALIGN 64)

string(LENGTH "${RAJA_PERFSUITE_GPU_BLOCKSIZES}" BLOCKSIZES_LENGTH)
if (BLOCKSIZES_LENGTH GREATER 0)
  message(STATUS "Using gpu block size(s): ${RAJA_PERFSUITE_GPU_BLOCKSIZES}")
else()
  message(STATUS "Using default gpu block size(s)")
endif()

# exclude RAJA make targets from top-level build...
add_subdirectory(tpl/RAJA)

get_property(RAJA_INCLUDE_DIRS DIRECTORY tpl/RAJA PROPERTY INCLUDE_DIRECTORIES)
include_directories(${RAJA_INCLUDE_DIRS})

set(CAMP_ENABLE_TESTS Off CACHE BOOL "")

if (ENABLE_RAJA_SEQUENTIAL)
  add_definitions(-DRUN_RAJA_SEQ)
endif ()
if (ENABLE_OPENMP)
  add_definitions(-DRUN_OPENMP)
endif ()
if (ENABLE_STDPAR)
  add_definitions(-DRUN_STDPAR)
endif ()

set(RAJA_PERFSUITE_VERSION_MAJOR 0)
set(RAJA_PERFSUITE_VERSION_MINOR 11)
set(RAJA_PERFSUITE_VERSION_PATCHLEVEL 0)

set(RAJA_PERFSUITE_DEPENDS RAJA)

if (RAJA_PERFSUITE_ENABLE_MPI)
  list(APPEND RAJA_PERFSUITE_DEPENDS mpi)
endif()
if (ENABLE_OPENMP)
  list(APPEND RAJA_PERFSUITE_DEPENDS openmp)
endif()
if (ENABLE_CUDA)
  list(APPEND RAJA_PERFSUITE_DEPENDS cuda)
endif()

# Kokkos requires hipcc as the CMAKE_CXX_COMPILER for HIP AMD/VEGA GPU
# platforms, whereas RAJAPerf Suite uses blt/CMake FindHIP to set HIP compiler. 
# Separate RAJAPerf Suite and Kokkos handling of HIP compilers

if ((ENABLE_HIP) AND (NOT ENABLE_KOKKOS))
  message(STATUS "HIP version: ${hip_VERSION}")
  if("${hip_VERSION}" VERSION_LESS "3.5")
    message(FATAL_ERROR "Trying to use HIP/ROCm version ${hip_VERSION}. RAJA Perf Suite requires HIP/ROCm version 3.5 or newer. ")
  endif()
  list(APPEND RAJA_PERFSUITE_DEPENDS blt::hip)
  list(APPEND RAJA_PERFSUITE_DEPENDS blt::hip_runtime)
endif()

set(RAJAPERF_BUILD_SYSTYPE $ENV{SYS_TYPE})
set(RAJAPERF_BUILD_HOST $ENV{HOSTNAME})

if (ENABLE_CUDA)
  if (ENABLE_KOKKOS)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -restrict --extended-lambda --expt-relaxed-constexpr")
  else()
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -restrict -arch ${CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr")
  endif()

  set(RAJAPERF_COMPILER "${CUDA_NVCC_EXECUTABLE}")
  list(APPEND RAJAPERF_COMPILER ${CMAKE_CXX_COMPILER})
  set(RAJAPERF_COMPILER_OPTIONS "${CUDA_NVCC_FLAGS}")
elseif (ENABLE_HIP)
  set(RAJAPERF_COMPILER "${HIP_HIPCC_EXECUTABLE}")
  list(APPEND RAJAPERF_COMPILER ${CMAKE_CXX_COMPILER})
  set(RAJAPERF_COMPILER_OPTIONS "${HIP_HIPCC_FLAGS}")
else()
  set(RAJAPERF_COMPILER "${CMAKE_CXX_COMPILER}")
  string(TOUPPER ${CMAKE_BUILD_TYPE} RAJAPERF_BUILD_TYPE)
  set(RAJAPERF_COMPILER_OPTIONS "${CMAKE_CXX_FLAGS_${RAJAPERF_BUILD_TYPE}}")
  list(APPEND RAJAPERF_COMPILER_OPTIONS ${CMAKE_CXX_FLAGS})
endif()

configure_file(${CMAKE_SOURCE_DIR}/src/rajaperf_config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/rajaperf_config.hpp)

include_directories($<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

# Make sure RAJA flags propagate (we need to do some tidying to
# remove project-specific CMake variables that are no longer needed)
set (CUDA_NVCC_FLAGS ${RAJA_NVCC_FLAGS})

#
# Each directory in the perf suite has its own CMakeLists.txt file.

# ENABLE_KOKKOS is A RAJAPerf Suite Option
if(ENABLE_KOKKOS)
  add_definitions(-DRUN_KOKKOS)
  if(ENABLE_HIP)
    set(Kokkos_ENABLE_HIP ON CACHE BOOL "Kokkos builds for AMD HIP set the
Kokkos_ENABLE_HIP variable to ON")
  endif()

  if(ENABLE_TARGET_OPENMP)
    set(Kokkos_ENABLE_OPENMPTARGET ON CACHE BOOL "Docstring")
    if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
      if(NOT EXPERIMENTAL_BUILD)
      message(FATAL_ERROR "Kokkos builds with OpenMPTarget require a Debug build to succeed at the moment. Rebuild with CMAKE_BUILD_TYPE=Debug. If you're a compiler developer, rebuild with -DEXPERIMENTAL_BUILD=ON")
      endif()
    endif()
  endif()

# ENABLE_CUDA IS A RAJA PERFSUITE OPTION
  if(ENABLE_CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Docstring")
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Docstring")
    enable_language(CUDA)
  endif()
  if(ENABLE_OPENMP)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Docstring")
  endif()

  add_subdirectory(tpl/kokkos)
  get_property(KOKKOS_INCLUDE_DIRS DIRECTORY tpl/kokkos PROPERTY INCLUDE_DIRECTORIES)
  include_directories(${KOKKOS_INCLUDE_DIRS})
  list(APPEND RAJA_PERFSUITE_DEPENDS kokkos)
endif()

add_subdirectory(src)

if (RAJA_PERFSUITE_ENABLE_TESTS)
  add_subdirectory(test)
endif()
