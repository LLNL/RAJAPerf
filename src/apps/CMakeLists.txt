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
  NAME apps
  SOURCES AppsData.cpp
          PRESSURE.cpp 
          PRESSURE-Cuda.cpp 
          PRESSURE-OMPTarget.cpp 
          ENERGY.cpp 
          ENERGY-Cuda.cpp 
          ENERGY-OMPTarget.cpp 
          VOL3D.cpp 
          VOL3D-Cuda.cpp 
          VOL3D-OMPTarget.cpp 
          DEL_DOT_VEC_2D.cpp 
          DEL_DOT_VEC_2D-Cuda.cpp 
          DEL_DOT_VEC_2D-OMPTarget.cpp 
          FIR.cpp
          FIR-Cuda.cpp
          FIR-OMPTarget.cpp
          LTIMES.cpp
          LTIMES_NOVIEW.cpp
          WIP-COUPLE.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
