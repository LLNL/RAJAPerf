raja_add_library(
  NAME stream
  SOURCES COPY.cpp 
          MUL.cpp 
          ADD.cpp 
          TRIAD.cpp 
          DOT.cpp 
  DEPENDS_ON common RAJA)
