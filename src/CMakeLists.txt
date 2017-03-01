include_directories(.)

add_subdirectory(common)
add_subdirectory(basic)
#add_subdirectory(livloops)
#add_subdirectory(polybench)
#add_subdirectory(stream)
#add_subdirectory(apps)

add_executable(raja-perf.exe
  RAJAPerfSuiteDriver.cxx) 

target_link_libraries(raja-perf.exe
  common
  basic
##  livloops
##  polybench
##  stream
##  apps
  )

if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(PERFSUITE_COMP_OPTIONS "${CMAKE_CXX_FLAGS_DEBUG}")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  set(PERFSUITE_COMP_OPTIONS "${CMAKE_CXX_FLAGS_RELEASE}")
elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(PERFSUITE_COMP_OPTIONS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()
list(APPEND PERFSUITE_COMP_OPTIONS ${CMAKE_CXX_FLAGS})

configure_file(${CMAKE_SOURCE_DIR}/extra/performance/src/rajaperf_build_info.hxx.in
  ${CMAKE_CURRENT_BINARY_DIR}/rajaperf_build_info.hxx)
