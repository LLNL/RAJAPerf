###############################################################################
# Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME stream-kokkos
  SOURCES ADD-Kokkos.cpp 
          COPY-Kokkos.cpp 
          DOT-Kokkos.cpp 
          MUL-Kokkos.cpp 
          TRIAD-Kokkos.cpp
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../stream
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
