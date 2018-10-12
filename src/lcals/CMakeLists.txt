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
# For details about use and distribution, please read RAJAPerf/LICENSE.
#
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
          INT_PREDICT.cpp 
          INT_PREDICT-Cuda.cpp 
          INT_PREDICT-OMPTarget.cpp 
          PLANCKIAN.cpp 
          PLANCKIAN-Cuda.cpp 
          PLANCKIAN-OMPTarget.cpp 
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
