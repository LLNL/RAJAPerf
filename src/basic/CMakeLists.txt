add_library(basic
  MULADDSUB.cxx)

target_link_libraries(basic
  common
  RAJA)
