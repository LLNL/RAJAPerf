###############################################################################
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
          TRAP_INT.cpp 
          INIT3.cpp
          INIT3-Cuda.cpp
          INIT3-OMPTarget.cpp
          REDUCE3_INT.cpp
          NESTED_INIT.cpp
          INIT_VIEW1D.cpp
          INIT_VIEW1D_OFFSET.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
