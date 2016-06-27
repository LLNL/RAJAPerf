project(RajaPerfSuite CXX)

find_package(RAJA REQUIRED)
include_directories(${RAJA_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 14)

#if(RAJA_USE_OPENMP)
  find_package(OpenMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  #endif()

include(ExternalProject)
# Set default ExternalProject root directory
SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/tpl)

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

enable_testing()

add_subdirectory(src)
