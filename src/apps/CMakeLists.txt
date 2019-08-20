###############################################################################
# Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME apps
  SOURCES AppsData.cpp
          WIP-COUPLE.cpp
          DEL_DOT_VEC_2D.cpp 
          DEL_DOT_VEC_2D-Cuda.cpp 
          DEL_DOT_VEC_2D-OMPTarget.cpp 
          ENERGY.cpp 
          ENERGY-Cuda.cpp 
          ENERGY-OMPTarget.cpp 
          FIR.cpp
          FIR-Cuda.cpp
          FIR-OMPTarget.cpp
          LTIMES.cpp
          LTIMES-Cuda.cpp
          LTIMES-OMPTarget.cpp
          LTIMES_NOVIEW.cpp
          LTIMES_NOVIEW-Cuda.cpp
          LTIMES_NOVIEW-OMPTarget.cpp
          PRESSURE.cpp 
          PRESSURE-Cuda.cpp 
          PRESSURE-OMPTarget.cpp 
          VOL3D.cpp 
          VOL3D-Cuda.cpp 
          VOL3D-OMPTarget.cpp 
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
