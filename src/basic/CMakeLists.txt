raja_add_library(
  NAME basic
  SOURCES MULADDSUB.cpp 
          IF_QUAD.cpp 
          TRAP_INT.cpp 
  DEPENDS_ON common RAJA)
