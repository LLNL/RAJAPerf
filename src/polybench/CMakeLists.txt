raja_add_library(
  NAME polybench
  SOURCES POLYBENCH_2MM.cpp 
          POLYBENCH_3MM.cpp
          #          POLYBENCH_ADI.cpp
  DEPENDS_ON common RAJA)

