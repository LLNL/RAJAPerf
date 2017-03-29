raja_add_library(
  NAME common
  SOURCES DataUtils.cxx Executor.cxx KernelBase.cxx OutputUtils.cxx RAJAPerfSuite.cxx RunParams.cxx
  DEPENDS_ON common RAJA)
