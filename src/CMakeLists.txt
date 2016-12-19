include_directories(.)

add_subdirectory(common)
add_subdirectory(basic)
#add_subdirectory(lloops)
#add_subdirectory(polybench)
#add_subdirectory(stream)
#add_subdirectory(apps)

add_executable(raja-perf.exe
  RAJAPerfSuiteDriver.cxx) 

target_link_libraries(raja-perf.exe
  common
  basic
##  lloops
##  polybench
##  stream
##  apps
  )
