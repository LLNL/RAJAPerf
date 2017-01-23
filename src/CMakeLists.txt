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

##file(WRITE  ./rajaperf_build_info.txt "Compiler: ${CMAKE_CXX_COMPILER}\n")
##file(APPEND ./rajaperf_build_info.txt "Compiler Flags: ${CMAKE_CXX_FLAGS}")

##add_custom_command(TARGET raja-perf.exe POST_BUILD 
##                   COMMAND COMMAND ${CMAKE_COMMAND} -E copy
##                   ${CMAKE_SOURCE_DIR}/src/rajaperf_build_info.txt
##                   ${CMAKE_CURRENT_BINARY_DIR}/rajaperf_build_info.txt)
