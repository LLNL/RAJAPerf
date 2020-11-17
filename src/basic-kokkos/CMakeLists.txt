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
          ATOMIC_PI-KokkosSeq.cpp
          ATOMIC_PI-KokkosCuda.cpp
          ATOMIC_PI-KokkosOMP.cpp
          ATOMIC_PI-KokkosOMPTarget.cpp
          DAXPY-KokkosSeq.cpp
          DAXPY-KokkosCuda.cpp
          DAXPY-KokkosOMP.cpp
          DAXPY-KokkosOMPTarget.cpp
          IF_QUAD-KokkosSeq.cpp
          IF_QUAD-KokkosCuda.cpp
          IF_QUAD-KokkosOMP.cpp
          IF_QUAD-KokkosOMPTarget.cpp
          INIT3-KokkosSeq.cpp
          INIT3-KokkosCuda.cpp
          INIT3-KokkosOMP.cpp
          INIT3-KokkosOMPTarget.cpp
          INIT_VIEW1D-KokkosSeq.cpp
          INIT_VIEW1D-KokkosCuda.cpp
          INIT_VIEW1D-KokkosOMP.cpp
          INIT_VIEW1D-KokkosOMPTarget.cpp
          INIT_VIEW1D_KokkosOFFSET-Seq.cpp
          INIT_VIEW1D_KokkosOFFSET-Cuda.cpp
          INIT_VIEW1D_KokkosOFFSET-OMP.cpp
          INIT_VIEW1D_KokkosOFFSET-OMPTarget.cpp
          MULADDSUB-KokkosSeq.cpp
          MULADDSUB-KokkosCuda.cpp
          MULADDSUB-KokkosOMP.cpp
          MULADDSUB-KokkosOMPTarget.cpp
          NESTED_INIT-KokkosSeq.cpp
          NESTED_INIT-KokkosCuda.cpp
          NESTED_INIT-KokkosOMP.cpp
          NESTED_INIT-KokkosOMPTarget.cpp
          REDUCE3_INT-KokkosSeq.cpp
          REDUCE3_INT-KokkosCuda.cpp
          REDUCE3_INT-KokkosOMP.cpp
          REDUCE3_INT-KokkosOMPTarget.cpp
          TRAP_INT-KokkosSeq.cpp
          TRAP_INT-KokkosCuda.cpp
          TRAP_INT-KokkosOMPTarget.cpp
          TRAP_INT-KokkosOMP.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )

