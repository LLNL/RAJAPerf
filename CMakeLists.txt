cmake_minimum_required(VERSION 3.1)

project(RajaPerfSuite CXX)

include(ExternalProject)
SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/tpl)

#find_package(RAJA REQUIRED)
#include_directories(${RAJA_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 14)

#if(RAJA_USE_OPENMP)
  find_package(OpenMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  #endif()

ExternalProject_Add(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git 
    GIT_TAG release-1.7.0
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON)

ExternalProject_Get_Property(googletest source_dir)
include_directories(${source_dir}/include)

ExternalProject_Get_Property(googletest binary_dir)
add_library(gtest      UNKNOWN IMPORTED)
add_library(gtest_main UNKNOWN IMPORTED)
set_target_properties(gtest PROPERTIES
  IMPORTED_LOCATION ${binary_dir}/libgtest.a
)
set_target_properties(gtest_main PROPERTIES
  IMPORTED_LOCATION ${binary_dir}/libgtest_main.a
)
add_dependencies(gtest      googletest)
add_dependencies(gtest_main googletest)

ExternalProject_Add(
    raja
    GIT_REPOSITORY https://github.com/LLNL/RAJA.git 
    GIT_TAG develop
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_BUILD_TYPE=Release
      -DRAJA_ENABLE_NESTED=On
      -DRAJA_ENABLE_TESTS=Off
      -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/tpl
      #INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON)

ExternalProject_Get_Property(raja install_dir)
message(STATUS ${install_dir})
set(RAJA_DIR ${install_dir}/share/cmake/raja/)
find_package(RAJA REQUIRED)
include_directories(${RAJA_INCLUDE_DIR})

# include_directories(${install_dir}/include)
# 
# add_library(RAJA      UNKNOWN IMPORTED)
# set_target_properties(RAJA PROPERTIES
#   IMPORTED_LOCATION ${install_dir}/lib/libRAJA.a
# )
# add_dependencies(RAJA      raja)


enable_testing()

add_subdirectory(src)
