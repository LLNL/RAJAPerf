###############################################################################
# Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

project(RAJAPerfSuite CXX)

cmake_minimum_required(VERSION 3.9)

option(ENABLE_RAJA_SEQUENTIAL "Run sequential variants of RAJA kernels. Disable
this, and all other variants, to run _only_ raw C loops." On)

#
# Note: the BLT build system is inheritted by RAJA and is initialized by RAJA
#

if (PERFSUITE_ENABLE_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()

set(ENABLE_TESTS Off CACHE BOOL "Enable BLT and RAJA tests")
set(ENABLE_EXAMPLES Off CACHE BOOL "Enable RAJA examples")
set(RAJA_ENABLE_EXERCISES Off CACHE BOOL "Enable RAJA exercises")

set(CMAKE_CXX_STANDARD 14)
set(BLT_CXX_STD c++14)

include(blt/SetupBLT.cmake)

#
# Define RAJA PERFSUITE settings...
#

cmake_dependent_option(RAJA_PERFSUITE_ENABLE_MPI "Build with MPI" On "ENABLE_MPI" Off)
cmake_dependent_option(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN "Build OpenMP scan variants" Off "ENABLE_OPENMP" Off)

#
# Define RAJA settings...
#

set(RAJA_ENABLE_TESTS Off CACHE BOOL "")
set(RAJA_ENABLE_EXAMPLES Off CACHE BOOL "")
set(ENABLE_DOCUMENTATION Off CACHE BOOL "")

set(ENABLE_TBB Off CACHE BOOL "")

set(RAJA_USE_CHRONO On CACHE BOOL "")

set(RAJA_RANGE_ALIGN 4)
set(RAJA_RANGE_MIN_LENGTH 32)
set(RAJA_DATA_ALIGN 64)


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
if (ENABLE_HIP)
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
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -restrict -arch ${CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr")

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

# Make sure RAJA flag propagate (we need to do some house cleaning to
# remove project-specific CMake variables that are no longer needed)
set (CUDA_NVCC_FLAGS ${RAJA_NVCC_FLAGS})

#
# Each directory in the perf suite has its own CMakeLists.txt file.
#
add_subdirectory(src)
