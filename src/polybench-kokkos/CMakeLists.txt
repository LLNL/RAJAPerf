###############################################################################
# Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../polybench)

blt_add_library(
  NAME polybench-kokkos
  SOURCES POLYBENCH_2MM-Kokkos.cpp
          POLYBENCH_3MM-Kokkos.cpp
          POLYBENCH_ADI-Kokkos.cpp
          POLYBENCH_ATAX-Kokkos.cpp
          POLYBENCH_FDTD_2D-Kokkos.cpp
          POLYBENCH_FLOYD_WARSHALL-Kokkos.cpp
          POLYBENCH_GEMM-Kokkos.cpp
          POLYBENCH_GEMVER-Kokkos.cpp
          POLYBENCH_GESUMMV-Kokkos.cpp
          POLYBENCH_HEAT_3D-Kokkos.cpp
          POLYBENCH_JACOBI_1D-Kokkos.cpp
          POLYBENCH_JACOBI_2D-Kokkos.cpp
          POLYBENCH_MVT-Kokkos.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
