# add_executable(copy
#   copy.cxx)
# 
# raja_add_benchmark(copy)

add_executable(add
  add-main.cxx)

#raja_add_benchmark(add)

# add_executable(mul
#   mul.cxx)
# 
# raja_add_benchmark(mul)
# 
# add_executable(triad
#   triad.cxx)
# 
# raja_add_benchmark(triad)
# 
# target_link_libraries(add ${RT_LIBRARIES} benchmark RAJA)
# 
# target_link_libraries(copy ${RT_LIBRARIES} benchmark RAJA)
# 
# target_link_libraries(mul ${RT_LIBRARIES} benchmark RAJA)
# 
# target_link_libraries(triad ${RT_LIBRARIES} benchmark RAJA)
