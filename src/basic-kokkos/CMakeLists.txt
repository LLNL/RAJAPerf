###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME basic-kokkos
  SOURCES 
          DAXPY-KokkosSeq.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )

