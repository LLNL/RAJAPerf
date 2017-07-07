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
# Configure RAJA
#

set(RAJA_ENABLE_TESTS Off CACHE BOOL "")
set(RAJA_ENABLE_EXAMPLES Off CACHE BOOL "")
set(RAJA_ENABLE_DOCUMENTATION Off CACHE BOOL "")

set(RAJA_USE_CHRONO On CACHE BOOL "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")
set(RAJA_COHERENCE_BLOCK_SIZE 64 CACHE INT "")

if(ENABLE_OPENMP)
set(RAJA_ENABLE_OPENMP On CACHE BOOL "")
else()
set(RAJA_ENABLE_OPENMP Off CACHE BOOL "")
endif()

if(ENABLE_CUDA)
set(RAJA_ENABLE_CUDA On CACHE BOOL "")
else()
set(RAJA_ENABLE_CUDA Off CACHE BOOL "")
endif()

# exclude RAJA make targets from top-level build...
add_subdirectory(tpl/RAJA EXCLUDE_FROM_ALL)

get_property(RAJA_INCLUDE_DIRS DIRECTORY tpl/RAJA PROPERTY INCLUDE_DIRECTORIES)
include_directories(${RAJA_INCLUDE_DIRS})


#
# Initialize the BLT build system
#

include(blt/SetupBLT.cmake)


#
# Setup variables to pass to Perf suite
#

#set(RAJA_PERFSUITE_VERSION "0.1.0")

string(TOUPPER ${CMAKE_BUILD_TYPE} RAJAPERF_BUILD_TYPE)
set(RAJAPERF_COMPILER_OPTIONS "${CMAKE_CXX_FLAGS_${RAJAPERF_BUILD_TYPE}}")
list(APPEND RAJAPERF_COMPILER_OPTIONS ${CMAKE_CXX_FLAGS})

##set(RAJAPERF_SYSTYPE $ENV{SYS_TYPE})
configure_file(${CMAKE_SOURCE_DIR}/src/rajaperf_build_info.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/bin/rajaperf_build_info.hpp)


#set(COMMON_PERFSUITE_DEPS RAJA)

#if (${RAJA_ENABLE_OPENMP})
#if (RAJA_ENABLE_OPENMP)
#list(APPEND COMMON_PERFSUITE_DEPS openmp)
#endif()

add_subdirectory(src)
