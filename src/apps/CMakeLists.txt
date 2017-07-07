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

set(rajaperf_apps_depends
    common
    RAJA)
if(ENABLE_OPENMP)
  list(APPEND rajaperf_apps_depends openmp)
endif()

blt_add_library(
  NAME apps
  SOURCES PRESSURE.cpp 
          ENERGY.cpp 
          VOL3D.cpp 
          DEL_DOT_VEC_2D.cpp 
          COUPLE.cpp
          FIR.cpp
  DEPENDS_ON ${rajaperf_apps_depends}
  )
