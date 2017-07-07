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

set(rajaperf_common_depends
    RAJA)
if(ENABLE_OPENMP)
  list(APPEND rajaperf_common_depends openmp)
endif()

blt_add_library(
  NAME common
  SOURCES DataUtils.cpp 
          Executor.cpp 
          KernelBase.cpp 
          OutputUtils.cpp 
          RAJAPerfSuite.cpp 
          RunParams.cpp
  DEPENDS_ON ${rajaperf_common_depends}
  )
