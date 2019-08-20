###############################################################################
# Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME basic
  SOURCES DAXPY.cpp
          DAXPY-Cuda.cpp 
          DAXPY-OMPTarget.cpp 
          IF_QUAD.cpp 
          IF_QUAD-Cuda.cpp 
          IF_QUAD-OMPTarget.cpp 
          INIT3.cpp
          INIT3-Cuda.cpp
          INIT3-OMPTarget.cpp
          INIT_VIEW1D.cpp
          INIT_VIEW1D-Cuda.cpp
          INIT_VIEW1D-OMPTarget.cpp
          INIT_VIEW1D_OFFSET.cpp
          INIT_VIEW1D_OFFSET-Cuda.cpp
          INIT_VIEW1D_OFFSET-OMPTarget.cpp
          MULADDSUB.cpp
          MULADDSUB-Cuda.cpp 
          MULADDSUB-OMPTarget.cpp 
          NESTED_INIT.cpp
          NESTED_INIT-Cuda.cpp
          NESTED_INIT-OMPTarget.cpp
          REDUCE3_INT.cpp
          REDUCE3_INT-Cuda.cpp
          REDUCE3_INT-OMPTarget.cpp
          TRAP_INT.cpp 
          TRAP_INT-Cuda.cpp 
          TRAP_INT-OMPTarget.cpp 
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
