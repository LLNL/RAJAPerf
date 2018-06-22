###############################################################################
# Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-738930
#
# All rights reserved.
#
# This file is part of the RAJA Performance Suite.
#
# For details about use and distribution, please read raja-perfsuite/LICENSE.
#
###############################################################################

cmake_minimum_required(VERSION 3.3)

project(RAJAPerfSuite CXX)

#
# Initialize the BLT build system
#

if (PERFSUITE_ENABLE_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()

set(ENABLE_TESTS On CACHE BOOL "")

include(blt/SetupBLT.cmake)

set(CMAKE_CXX_STANDARD 11)
set(BLT_CXX_STANDARD 11)

#
# Define RAJA settings...
#

set(ENABLE_TESTS On CACHE Bool "")
set(ENABLE_EXAMPLES On CACHE Bool "")
set(ENABLE_DOCUMENTATION Off CACHE Bool "")

set(ENABLE_TBB Off CACHE Bool "")

set(RAJA_USE_CHRONO On CACHE Bool "")

set(RAJA_RANGE_ALIGN 4)
set(RAJA_RANGE_MIN_LENGTH 32)
set(RAJA_DATA_ALIGN 64)
set(RAJA_COHERENCE_BLOCK_SIZE 64)


# exclude RAJA make targets from top-level build...
add_subdirectory(tpl/RAJA)

get_property(RAJA_INCLUDE_DIRS DIRECTORY tpl/RAJA PROPERTY INCLUDE_DIRECTORIES)
include_directories(${RAJA_INCLUDE_DIRS})


#
# Setup variables to pass to Perf suite
#

set(RAJA_PERFSUITE_VERSION_MAJOR 0)
set(RAJA_PERFSUITE_VERSION_MINOR 3)
set(RAJA_PERFSUITE_VERSION_PATCHLEVEL 0)

set(RAJA_PERFSUITE_DEPENDS RAJA)
if (ENABLE_OPENMP)
  list(APPEND RAJA_PERFSUITE_DEPENDS openmp)
endif()
if (ENABLE_CUDA)
  list(APPEND RAJA_PERFSUITE_DEPENDS cuda)
endif() 

set(RAJAPERF_BUILD_SYSTYPE $ENV{SYS_TYPE})
set(RAJAPERF_BUILD_HOST $ENV{HOSTNAME})

if (ENABLE_CUDA)
set(RAJAPERF_COMPILER "${CUDA_NVCC_EXECUTABLE}")
list(APPEND RAJAPERF_COMPILER ${CMAKE_CXX_COMPILER})
set(RAJAPERF_COMPILER_OPTIONS "${CUDA_NVCC_FLAGS}")
else()
set(RAJAPERF_COMPILER "${CMAKE_CXX_COMPILER}")
string(TOUPPER ${CMAKE_BUILD_TYPE} RAJAPERF_BUILD_TYPE)
set(RAJAPERF_COMPILER_OPTIONS "${CMAKE_CXX_FLAGS_${RAJAPERF_BUILD_TYPE}}")
list(APPEND RAJAPERF_COMPILER_OPTIONS ${CMAKE_CXX_FLAGS})
endif()

configure_file(${CMAKE_SOURCE_DIR}/src/rajaperf_config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/bin/rajaperf_config.hpp)

# Make sure RAJA flag propagate
set (CUDA_NVCC_FLAGS ${RAJA_NVCC_FLAGS})

#
# Each directory in the perf suite has its own CMakeLists.txt file.
#
add_subdirectory(src)
