raja_add_library(
  NAME basic
  SOURCES MULADDSUB.cpp 
          IF_QUAD.cpp 
          TRAP_INT.cpp 
          INIT3.cpp
          REDUCE3_INT.cpp
          NESTED_INIT.cpp
  DEPENDS_ON common RAJA)
