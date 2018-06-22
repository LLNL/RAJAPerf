###############################################################################
# Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-738930
#
# All rights reserved.
#
# This file is part of the RAJA Performance Suite.
#
# For details about use and distribution, please read raja-perfsuite/LICENSE.
#
###############################################################################

blt_add_library(
  NAME basic
  SOURCES MULADDSUB.cpp
          MULADDSUB-Cuda.cpp 
          MULADDSUB-OMPTarget.cpp 
          IF_QUAD.cpp 
          IF_QUAD-Cuda.cpp 
          IF_QUAD-OMPTarget.cpp 
          TRAP_INT.cpp 
          TRAP_INT-Cuda.cpp 
          TRAP_INT-OMPTarget.cpp 
          INIT3.cpp
          INIT3-Cuda.cpp
          INIT3-OMPTarget.cpp
          REDUCE3_INT.cpp
          REDUCE3_INT-Cuda.cpp
          REDUCE3_INT-OMPTarget.cpp
          NESTED_INIT.cpp
          NESTED_INIT-Cuda.cpp
          NESTED_INIT-OMPTarget.cpp
          INIT_VIEW1D.cpp
          INIT_VIEW1D-Cuda.cpp
          INIT_VIEW1D-OMPTarget.cpp
          INIT_VIEW1D_OFFSET.cpp
          INIT_VIEW1D_OFFSET-Cuda.cpp
          INIT_VIEW1D_OFFSET-OMPTarget.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
