###############################################################################
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-xxxxxx
#
# All rights reserved.
#
# This file is part of the RAJA Performance Suite.
#
# For more information, please see the file LICENSE in the top-level directory.
#
###############################################################################

cmake_minimum_required(VERSION 3.1)

project(RAJAPerfSuite CXX)

#
# Initialize the BLT build system
#

if (PERFSUITE_ENABLE_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()

set(ENABLE_TESTS On CACHE BOOL "")

include(blt/SetupBLT.cmake)

#
# Define RAJA default settings...
#

#
# Don't turn this off or code won't build (for now)....
#
set(RAJA_BUILD_WITH_BLT On)

set(RAJA_ENABLE_TESTS On)
set(RAJA_ENABLE_EXAMPLES Off)
set(RAJA_ENABLE_DOCUMENTATION Off)

set(RAJA_USE_CHRONO On)

set(RAJA_RANGE_ALIGN 4)
set(RAJA_RANGE_MIN_LENGTH 32)
set(RAJA_DATA_ALIGN 64)
set(RAJA_COHERENCE_BLOCK_SIZE 64)

if (ENABLE_OPENMP)
  set(RAJA_ENABLE_OPENMP On)
else()
  set(RAJA_ENABLE_OPENMP Off)
endif()

if (ENABLE_CUDA)
  set(RAJA_ENABLE_CUDA On)
else()
  set(RAJA_ENABLE_CUDA Off)
endif()

# exclude RAJA make targets from top-level build...
add_subdirectory(tpl/RAJA)

get_property(RAJA_INCLUDE_DIRS DIRECTORY tpl/RAJA PROPERTY INCLUDE_DIRECTORIES)
include_directories(${RAJA_INCLUDE_DIRS})


#
# Setup variables to pass to Perf suite
#

set(RAJA_PERFSUITE_DEPENDS RAJA)
if (RAJA_ENABLE_OPENMP)
  list(APPEND RAJA_PERFSUITE_DEPENDS openmp)
endif()
if (ENABLE_CUDA)
  list(APPEND RAJA_PERFSUITE_DEPENDS cuda)
endif() 

#set(RAJA_PERFSUITE_VERSION "0.1.0")

string(TOUPPER ${CMAKE_BUILD_TYPE} RAJAPERF_BUILD_TYPE)
set(RAJAPERF_COMPILER_OPTIONS "${CMAKE_CXX_FLAGS_${RAJAPERF_BUILD_TYPE}}")
list(APPEND RAJAPERF_COMPILER_OPTIONS ${CMAKE_CXX_FLAGS})

##set(RAJAPERF_SYSTYPE $ENV{SYS_TYPE})
configure_file(${CMAKE_SOURCE_DIR}/src/rajaperf_build_info.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/bin/rajaperf_build_info.hpp)


#
# Each directory in the perf suite has its own CMakeLists.txt file.
#
add_subdirectory(src)
