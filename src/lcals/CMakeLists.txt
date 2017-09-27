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
          INT_PREDICT.cpp 
          DIFF_PREDICT.cpp 
          FIRST_DIFF.cpp 
          PLANCKIAN.cpp 
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
