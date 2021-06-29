###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################


include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../apps)

blt_add_library(
  NAME apps
  SOURCES AppsData.cpp
          DEL_DOT_VEC_2D-Kokkos.cpp 
          ENERGY-Kokkos.cpp
          FIR-Kokkos.cpp
          HALOEXCHANGE-Kokkos.cpp
          LTIMES-Kokkos.cpp
          LTIMES_NOVIEW-Kokkos.cpp
          PRESSURE-Kokkos.cpp 
          VOL3D-Kokkos.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
