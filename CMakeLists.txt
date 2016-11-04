cmake_minimum_required(VERSION 3.1)

project(RajaPerfSuite CXX)

include(cmake/macros.cmake)

ExternalProject_Add(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG master
    CMAKE_ARGS                
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CXX_LINK_FLAGS=${RT_LIBRARIES}
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON)

ExternalProject_Get_Property(googlebenchmark source_dir)
include_directories(${source_dir}/include)

ExternalProject_Get_Property(googlebenchmark binary_dir)
add_library(benchmark      UNKNOWN IMPORTED)
set_target_properties(benchmark PROPERTIES
  IMPORTED_LOCATION ${binary_dir}/src/libbenchmark.a
)
add_dependencies(benchmark      googlebenchmark)

add_subdirectory(src)
