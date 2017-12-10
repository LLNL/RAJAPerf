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
  NAME lcals
  SOURCES HYDRO_1D.cpp 
          EOS.cpp 
          EOS-Cuda.cpp 
          EOS-OMPTarget.cpp 
          INT_PREDICT.cpp 
          DIFF_PREDICT.cpp 
          DIFF_PREDICT-Cuda.cpp 
          DIFF_PREDICT-OMPTarget.cpp 
          FIRST_DIFF.cpp 
          FIRST_DIFF-Cuda.cpp 
          FIRST_DIFF-OMPTarget.cpp 
          PLANCKIAN.cpp 
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
