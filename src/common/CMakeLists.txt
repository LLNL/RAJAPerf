###############################################################################
# Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME common
  SOURCES DataUtils.cpp 
          Executor.cpp 
          KernelBase.cpp 
          OutputUtils.cpp 
          RAJAPerfSuite.cpp 
          RunParams.cpp
  INCLUDES ${PROJECT_BINARY_DIR}/include/
  DEPENDS_ON ${RAJA_PERFSUITE_DEPENDS}
  )
