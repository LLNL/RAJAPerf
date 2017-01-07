add_library(basic
  INIT3.cxx
  MULADDSUB.cxx
  IF_QUAD.cxx
  TRAP_INT.cxx)

target_link_libraries(basic
  common
  RAJA)
