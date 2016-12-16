include_directories(.)

add_subdirectory(common)
add_subdirectory(lcals)
#add_subdirectory(stream)

add_executable(raja-perf.exe
  RAJAPerfSuiteDriver.cxx) 

target_link_libraries(raja-perf.exe
  common
  lcals)
