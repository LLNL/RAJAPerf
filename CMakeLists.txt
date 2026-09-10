###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA Performance Suite.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

# C is required for googletest to find Threads
project(RAJAPerfSuite LANGUAGES CXX C)

cmake_minimum_required(VERSION 3.24)

# Detect C++ standard and add appropriate flag _before_ loading BLT
set(COMPILERS_KNOWN_TO_CMAKE33 AppleClang Clang GNU MSVC)

include(CheckCXXCompilerFlag)
if(NOT DEFINED BLT_CXX_STD)
  if("cxx_std_20" IN_LIST CMAKE_CXX_KNOWN_FEATURES)
    set(BLT_CXX_STD c++20 CACHE STRING "Version of C++ standard")
    message("Using C++ standard: ${BLT_CXX_STD}")
  elseif("${CMAKE_CXX_COMPILER_ID}" IN_LIST COMPILERS_KNOWN_TO_CMAKE33)
    set(BLT_CXX_STD c++20 CACHE STRING "Version of C++ standard")
    message("Using C++ standard: ${BLT_CXX_STD}")
  else() #cmake has no idea what to do, do it ourselves...
    set(flag_var "c++20")
    CHECK_CXX_COMPILER_FLAG("-std=${flag_var}" COMPILER_SUPPORTS_${flag_var})
    if(COMPILER_SUPPORTS_${flag_var})
      set(BLT_CXX_STD ${flag_var} CACHE STRING "Version of C++ standard")
      message("Using C++ standard: ${BLT_CXX_STD}")
    endif()
    unset(flag_var)
  endif()
else() #check BLT_CXX_STD is high enough by disallowing the only invalid option
  if(("${BLT_CXX_STD}" STREQUAL "c++98") OR
     ("${BLT_CXX_STD}" STREQUAL "c++11") OR
     ("${BLT_CXX_STD}" STREQUAL "c++14") OR
     ("${BLT_CXX_STD}" STREQUAL "c++17"))
    message(FATAL_ERROR "RAJA requires minimum C++ standard of c++20")
  endif()
endif(NOT DEFINED BLT_CXX_STD)

set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT BLT_LOADED)
  if (DEFINED BLT_SOURCE_DIR)
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
  else ()
    set (BLT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/blt CACHE PATH "")

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "\
      The BLT submodule is not present. \
      If in git repository run the following two commands:\n \
      git submodule init\n \
      git submodule update")
    endif ()
  endif ()

  include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
endif()

option(ENABLE_KOKKOS "Include Kokkos implementations of the kernels in the RAJA Perfsuite" Off)

#
# Define RAJA PERFSUITE settings...
#

option(ENABLE_RAJA_SEQUENTIAL "Run sequential variants of RAJA kernels. Disable
this, and all other variants, to run _only_ base variants." On)

if (PERFSUITE_ENABLE_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()

cmake_dependent_option(RAJA_PERFSUITE_ENABLE_TESTS "Enable RAJA Perf Suite Tests" On "ENABLE_TESTS" Off)

if (ENABLE_TESTS)

  set(RAJA_ENABLE_TESTS Off CACHE BOOL "")
  set(CAMP_ENABLE_TESTS Off CACHE BOOL "")

endif()

option(PERFSUITE_RUN_SHORT_TEST "Shorter test run to avoid timeout in some CI cases." Off)

cmake_dependent_option(RAJA_PERFSUITE_ENABLE_MPI "Build with MPI" On "ENABLE_MPI" Off)
if (RAJA_PERFSUITE_ENABLE_MPI)
set(RAJA_PERFSUITE_NUM_MPI_TASKS 4 CACHE STRING "Number of MPI tasks in tests")
else()
set(RAJA_PERFSUITE_NUM_MPI_TASKS 0 CACHE INTERNAL "Number of MPI tasks in tests")
endif()
message(STATUS "Using RAJA_PERFSUITE_NUM_MPI_TASKS: ${RAJA_PERFSUITE_NUM_MPI_TASKS}")

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

set(RAJA_PERFSUITE_TUNING_CUDA_ARCH "0" CACHE STRING "CUDA arch to tune the execution for, ex '700' for sm_70")
set(RAJA_PERFSUITE_TUNING_HIP_ARCH "0" CACHE STRING "HIP arch to tune the execution for, ex '910' for gfx90a, '942' for gfx942")

set(RAJA_PERFSUITE_GPU_BLOCKSIZES "" CACHE STRING "Comma separated list of GPU block sizes, ex '256,1024'")

set(RAJA_PERFSUITE_ATOMIC_REPLICATIONS "" CACHE STRING "Comma separated list of atomic replications, ex '1,256,4096'")

set(RAJA_PERFSUITE_GPU_ITEMS_PER_THREAD "" CACHE STRING "Comma separated list of items per thread for all GPU kernels, ex '1,2,4,8'")

set(RAJA_RANGE_ALIGN 4)
set(RAJA_RANGE_MIN_LENGTH 32)
set(RAJA_DATA_ALIGN 64)

string(LENGTH "${RAJA_PERFSUITE_TUNING_CUDA_ARCH}" CUDA_ARCH_LENGTH)
if (CUDA_ARCH_LENGTH GREATER 1)
  message(STATUS "Using cuda tunings for arch: ${RAJA_PERFSUITE_TUNING_CUDA_ARCH}")
else()
  message(STATUS "Using default cuda arch tunings")
endif()

string(LENGTH "${RAJA_PERFSUITE_TUNING_HIP_ARCH}" HIP_ARCH_LENGTH)
if (HIP_ARCH_LENGTH GREATER 1)
  message(STATUS "Using hip tunings for arch: ${RAJA_PERFSUITE_TUNING_HIP_ARCH}")
else()
  message(STATUS "Using default hip arch tunings")
endif()

string(LENGTH "${RAJA_PERFSUITE_GPU_BLOCKSIZES}" BLOCKSIZES_LENGTH)
if (BLOCKSIZES_LENGTH GREATER 0)
  message(STATUS "Using gpu block size(s): ${RAJA_PERFSUITE_GPU_BLOCKSIZES}")
else()
  message(STATUS "Using default gpu block size(s)")
endif()

string(LENGTH "${RAJA_PERFSUITE_ATOMIC_REPLICATIONS}" ATOMIC_REPLICATIONS_LENGTH)
if (ATOMIC_REPLICATIONS_LENGTH GREATER 0)
  message(STATUS "Using atomic replication(s): ${RAJA_PERFSUITE_ATOMIC_REPLICATIONS}")
else()
  message(STATUS "Using default atomic replication(s)")
endif()

string(LENGTH "${RAJA_PERFSUITE_GPU_ITEMS_PER_THREAD}" GPU_ITEMS_PER_THREAD_LENGTH)
if (GPU_ITEMS_PER_THREAD_LENGTH GREATER 0)
  message(STATUS "Using gpu items per thread(s): ${RAJA_PERFSUITE_GPU_ITEMS_PER_THREAD}")
else()
  message(STATUS "Using default gpu items per thread(s)")
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

if (PERFSUITE_RUN_SHORT_TEST)
  add_definitions(-DRUN_RAJAPERF_SHORT_TEST)
endif()

set(RAJA_PERFSUITE_VERSION_MAJOR 2025)
set(RAJA_PERFSUITE_VERSION_MINOR 12)
set(RAJA_PERFSUITE_VERSION_PATCHLEVEL 1)

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
if (RAJA_ENABLE_SYCL)
  list(APPEND RAJA_PERFSUITE_DEPENDS sycl)
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

#
# Are we using Caliper
#
set(RAJA_PERFSUITE_USE_CALIPER off CACHE BOOL "")
if (RAJA_PERFSUITE_USE_CALIPER)
  set(RAJA_PERFSUITE_USE_CALIPER_SUBKERNEL on CACHE BOOL
      "Enable Caliper subkernel regions")
  find_package(caliper REQUIRED)
  list(APPEND RAJA_PERFSUITE_DEPENDS caliper)
  add_definitions(-DRAJA_PERFSUITE_USE_CALIPER)
  message(STATUS "Using Caliper")
  if (RAJA_PERFSUITE_USE_CALIPER_SUBKERNEL)
    add_definitions(-DRAJA_PERFSUITE_USE_CALIPER_SUBKERNEL)
    message(STATUS "Using Caliper subkernel regions")
  endif()
  find_package(adiak REQUIRED)
  # use ${adiak_LIBRARIES} since version could have adiak vs adiak::adiak export
  list(APPEND RAJA_PERFSUITE_DEPENDS ${adiak_LIBRARIES})
  if (ENABLE_CUDA)
    # Adiak will propagate -pthread from spectrum mpi from a spack install of Caliper with +mpi; and needs to be handled even if RAJAPerf is non MPI program
    # We should delegate to BLT to handle unguarded -pthread from any dependencies, but currently BLT doesn't
    set_target_properties(${adiak_LIBRARIES} PROPERTIES INTERFACE_COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>;$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pthread>")
    # the following for adiak-0.2.2
    if (TARGET adiak::mpi)
      set_target_properties(adiak::mpi PROPERTIES INTERFACE_COMPILE_OPTIONS "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>;$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pthread>")
    endif ()
  endif ()
  message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
  message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")
  include_directories(${caliper_INCLUDE_DIR})
  include_directories(${adiak_INCLUDE_DIRS})
else ()
  unset(RAJA_PERFSUITE_USE_CALIPER_SUBKERNEL CACHE)
endif ()


set(RAJAPERF_BUILD_SYSTYPE $ENV{SYS_TYPE})
set(RAJAPERF_BUILD_HOST $ENV{HOSTNAME})

if (ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -restrict --extended-lambda --expt-relaxed-constexpr")

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

if (RAJA_PERFSUITE_ENABLE_DOCUMENTATION)
  add_subdirectory(docs)
endif ()
