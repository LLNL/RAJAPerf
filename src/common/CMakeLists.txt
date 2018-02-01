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
  NAME common
  SOURCES DataUtils.cpp 
          Executor.cpp 
          KernelBase.cpp 
          OutputUtils.cpp 
          RAJAPerfSuite.cpp 
          RunParams.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_DEPENDS}
  )
