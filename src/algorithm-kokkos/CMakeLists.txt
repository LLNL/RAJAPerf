###############################################################################
# Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../algorithm)


blt_add_library(
  NAME algorithm-kokkos
  SOURCES SORT-Kokkos.cpp
          SORTPAIRS-Kokkos.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
