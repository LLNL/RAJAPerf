add_executable(lcals-aloops
  lcals_aloops.cxx)

add_test(lcals-a
  lcals-aloops)

target_link_libraries(lcals-aloops RAJA gtest gtest_main)
