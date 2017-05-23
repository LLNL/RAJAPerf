raja_add_library(
  NAME polybench
  SOURCES POLYBENCH_2MM.cpp 
          POLYBENCH_3MM.cpp
  DEPENDS_ON common RAJA)

