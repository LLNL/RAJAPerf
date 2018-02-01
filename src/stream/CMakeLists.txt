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
  NAME stream
  SOURCES ADD.cpp
          ADD-Cuda.cpp
          ADD-OMPTarget.cpp
          COPY.cpp 
          COPY-Cuda.cpp
          COPY-OMPTarget.cpp
          DOT.cpp 
          DOT-Cuda.cpp 
          DOT-OMPTarget.cpp 
          MUL.cpp 
          MUL-Cuda.cpp 
          MUL-OMPTarget.cpp 
          TRIAD.cpp 
          TRIAD-Cuda.cpp 
          TRIAD-OMPTarget.cpp 
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
