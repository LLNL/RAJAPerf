add_executable(daxpy
  daxpy.cxx)

add_test(daxpy-test
  daxpy)

add_executable(ddot
  ddot.cxx)

add_test(ddot-test
  ddot)

target_link_libraries(daxpy ${RT_LIBRARIES} benchmark RAJA)

target_link_libraries(ddot ${RT_LIBRARIES} benchmark RAJA)
