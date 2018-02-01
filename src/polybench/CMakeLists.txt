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
  NAME polybench
  SOURCES POLYBENCH_2MM.cpp 
          POLYBENCH_2MM-Cuda.cpp
          POLYBENCH_2MM-OMPTarget.cpp
          POLYBENCH_3MM.cpp
          POLYBENCH_3MM-Cuda.cpp
          POLYBENCH_3MM-OMPTarget.cpp
          POLYBENCH_GEMMVER.cpp
          POLYBENCH_GEMMVER-Cuda.cpp
          POLYBENCH_GEMMVER-OMPTarget.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
