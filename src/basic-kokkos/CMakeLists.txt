###############################################################################
# Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../basic)

blt_add_library(
  NAME basic-kokkos
  SOURCES
          PI_ATOMIC-Kokkos.cpp
          ARRAY_OF_PTRS-Kokkos.cpp
          DAXPY-Kokkos.cpp
          IF_QUAD-Kokkos.cpp
          INIT3-Kokkos.cpp
          INIT_VIEW1D-Kokkos.cpp
          INIT_VIEW1D_OFFSET-Kokkos.cpp
          MULADDSUB-Kokkos.cpp
          NESTED_INIT-Kokkos.cpp
          REDUCE3_INT-Kokkos.cpp
          TRAP_INT-Kokkos.cpp
          DAXPY_ATOMIC-Kokkos.cpp
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../basic
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
