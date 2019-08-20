###############################################################################
# Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME lcals
  SOURCES DIFF_PREDICT.cpp 
          DIFF_PREDICT-Cuda.cpp 
          DIFF_PREDICT-OMPTarget.cpp 
          EOS.cpp 
          EOS-Cuda.cpp 
          EOS-OMPTarget.cpp 
          FIRST_DIFF.cpp 
          FIRST_DIFF-Cuda.cpp 
          FIRST_DIFF-OMPTarget.cpp 
          HYDRO_1D.cpp 
          HYDRO_1D-Cuda.cpp
          HYDRO_1D-OMPTarget.cpp
          HYDRO_2D.cpp 
          HYDRO_2D-Cuda.cpp
          HYDRO_2D-OMPTarget.cpp
          INT_PREDICT.cpp 
          INT_PREDICT-Cuda.cpp 
          INT_PREDICT-OMPTarget.cpp 
          PLANCKIAN.cpp 
          PLANCKIAN-Cuda.cpp 
          PLANCKIAN-OMPTarget.cpp 
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
