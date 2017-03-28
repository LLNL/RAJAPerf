raja_add_library(
  NAME basic
  SOURCES MULADDSUB.cxx IF_QUAD.cxx TRAP_INT.cxx
  DEPENDS_ON basic common RAJA)
