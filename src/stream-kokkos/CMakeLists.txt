###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../stream)

blt_add_library(
  NAME stream-kokkos
  SOURCES ADD-Kokkos.cpp 
          COPY-Kokkos.cpp 
          DOT-Kokkos.cpp 
          MUL-Kokkos.cpp 
          TRIAD-Kokkos.cpp 
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
