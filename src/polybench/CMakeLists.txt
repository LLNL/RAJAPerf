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
  NAME polybench
  SOURCES POLYBENCH_2MM.cpp 
          POLYBENCH_2MM-Cuda.cpp
          POLYBENCH_2MM-OMPTarget.cpp
          POLYBENCH_3MM.cpp
          POLYBENCH_3MM-Cuda.cpp
          POLYBENCH_3MM-OMPTarget.cpp
          POLYBENCH_ADI.cpp
          POLYBENCH_ADI-Cuda.cpp
          POLYBENCH_ADI-OMPTarget.cpp
          POLYBENCH_ATAX.cpp
          POLYBENCH_ATAX-Cuda.cpp
          POLYBENCH_ATAX-OMPTarget.cpp
          POLYBENCH_FDTD_2D.cpp
          POLYBENCH_FDTD_2D-Cuda.cpp
          POLYBENCH_FDTD_2D-OMPTarget.cpp
          POLYBENCH_GEMVER.cpp
          POLYBENCH_GEMVER-Cuda.cpp
          POLYBENCH_GEMVER-OMPTarget.cpp
          POLYBENCH_JACOBI_1D.cpp
          POLYBENCH_JACOBI_1D-Cuda.cpp
          POLYBENCH_JACOBI_1D-OMPTarget.cpp
          POLYBENCH_MVT.cpp
          POLYBENCH_MVT-Cuda.cpp
          POLYBENCH_MVT-OMPTarget.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
