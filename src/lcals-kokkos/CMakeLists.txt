###############################################################################
# Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################


blt_add_library(
  NAME lcals-kokkos
  SOURCES
          DIFF_PREDICT-Kokkos.cpp
          EOS-Kokkos.cpp
          FIRST_DIFF-Kokkos.cpp
          FIRST_MIN-Kokkos.cpp
          FIRST_SUM-Kokkos.cpp
          GEN_LIN_RECUR-Kokkos.cpp
          HYDRO_1D-Kokkos.cpp
          HYDRO_2D-Kokkos.cpp
          INT_PREDICT-Kokkos.cpp
          PLANCKIAN-Kokkos.cpp
          TRIDIAG_ELIM-Kokkos.cpp
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../lcals
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
