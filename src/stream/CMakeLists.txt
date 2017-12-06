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
  NAME stream
  SOURCES COPY.cpp 
          MUL.cpp 
          ADD.cpp 
          ADD-Cuda.cpp 
          ADD-OMPTarget.cpp 
          TRIAD.cpp 
          DOT.cpp 
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
