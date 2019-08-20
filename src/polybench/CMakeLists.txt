###############################################################################
# Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
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
          POLYBENCH_FLOYD_WARSHALL.cpp
          POLYBENCH_FLOYD_WARSHALL-Cuda.cpp
          POLYBENCH_FLOYD_WARSHALL-OMPTarget.cpp
          POLYBENCH_GEMM.cpp
          POLYBENCH_GEMM-Cuda.cpp
          POLYBENCH_GEMM-OMPTarget.cpp
          POLYBENCH_GEMVER.cpp
          POLYBENCH_GEMVER-Cuda.cpp
          POLYBENCH_GEMVER-OMPTarget.cpp
          POLYBENCH_GESUMMV.cpp
          POLYBENCH_GESUMMV-Cuda.cpp
          POLYBENCH_GESUMMV-OMPTarget.cpp
          POLYBENCH_HEAT_3D.cpp
          POLYBENCH_HEAT_3D-Cuda.cpp
          POLYBENCH_HEAT_3D-OMPTarget.cpp
          POLYBENCH_JACOBI_1D.cpp
          POLYBENCH_JACOBI_1D-Cuda.cpp
          POLYBENCH_JACOBI_1D-OMPTarget.cpp
          POLYBENCH_JACOBI_2D.cpp
          POLYBENCH_JACOBI_2D-Cuda.cpp
          POLYBENCH_JACOBI_2D-OMPTarget.cpp
          POLYBENCH_MVT.cpp
          POLYBENCH_MVT-Cuda.cpp
          POLYBENCH_MVT-OMPTarget.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
