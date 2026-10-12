###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA Performance Suite.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME polybench-kokkos
  SOURCES
          POLYBENCH_GEMM-Kokkos.cpp
          POLYBENCH_HEAT_3D-Kokkos.cpp
          POLYBENCH_JACOBI_1D-Kokkos.cpp
          POLYBENCH_JACOBI_2D-Kokkos.cpp
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../polybench
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
