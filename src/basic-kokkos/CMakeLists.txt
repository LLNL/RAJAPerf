###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../basic)

blt_add_library(
  NAME basic-kokkos
  SOURCES 
          DAXPY-KokkosSeq.cpp
	  DAXPY-KokkosOMP.cpp
	  DAXPY-KokkosCuda.cpp
	  ATOMIC_PI-KokkosOMP.cpp
	  ATOMIC_PI-KokkosSeq.cpp
	  ATOMIC_PI-KokkosCuda.cpp
	  IF_QUAD-KokkosSeq.cpp
	  IF_QUAD-KokkosOMP.cpp
	  IF_QUAD-KokkosCuda.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )

