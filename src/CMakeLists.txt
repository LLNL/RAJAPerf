include_directories(.)

add_subdirectory(common)
add_subdirectory(apps)
add_subdirectory(basic)
#add_subdirectory(livloops)
add_subdirectory(polybench)
#add_subdirectory(stream)

raja_add_executable(
  NAME raja-perf.exe
  SOURCES RAJAPerfSuiteDriver.cpp
  DEPENDS_ON 
  common
  apps
  basic
  RAJA
##  livloops
  polybench
##  stream
  )

if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(RAJAPERF_COMPILER_OPTIONS "${CMAKE_CXX_FLAGS_DEBUG}")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  set(RAJAPERF_COMPILER_OPTIONS "${CMAKE_CXX_FLAGS_RELEASE}")
elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(RAJAPERF_COMPILER_OPTIONS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()
list(APPEND RAJAPERF_COMPILER_OPTIONS ${CMAKE_CXX_FLAGS})

##set(RAJAPERF_SYSTYPE $ENV{SYS_TYPE})

configure_file(${CMAKE_SOURCE_DIR}/extra/performance/src/rajaperf_build_info.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/rajaperf_build_info.hpp)
