###############################################################################
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-xxxxxx
#
# All rights reserved.
#
# This file is part of the RAJA Performance Suite.
#
# For more information, please see the file LICENSE in the top-level directory.
#
###############################################################################

set(rajaperf_stream_depends
    common
    RAJA)
if(ENABLE_OPENMP)
  list(APPEND rajaperf_stream_depends openmp)
endif()

blt_add_library(
  NAME stream
  SOURCES COPY.cpp 
          MUL.cpp 
          ADD.cpp 
          TRIAD.cpp 
          DOT.cpp 
  DEPENDS_ON ${rajaperf_stream_depends}
  )
