raja_add_library(
  NAME common
  SOURCES DataUtils.cpp 
          Executor.cpp 
          KernelBase.cpp 
          OutputUtils.cpp 
          RAJAPerfSuite.cpp 
          RunParams.cpp
  DEPENDS_ON RAJA)
