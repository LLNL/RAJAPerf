###############################################################################
# Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME comm
  SOURCES HALOEXCHANGE_base.cpp
          HALOEXCHANGE.cpp
          HALOEXCHANGE-Seq.cpp
          HALOEXCHANGE-Hip.cpp
          HALOEXCHANGE-Cuda.cpp
          HALOEXCHANGE-OMP.cpp
          HALOEXCHANGE-OMPTarget.cpp
          HALOEXCHANGE_FUSED.cpp
          HALOEXCHANGE_FUSED-Seq.cpp
          HALOEXCHANGE_FUSED-Hip.cpp
          HALOEXCHANGE_FUSED-Cuda.cpp
          HALOEXCHANGE_FUSED-OMP.cpp
          HALOEXCHANGE_FUSED-OMPTarget.cpp
          MPI_HALOEXCHANGE.cpp
          MPI_HALOEXCHANGE-Seq.cpp
          MPI_HALOEXCHANGE-Hip.cpp
          MPI_HALOEXCHANGE-Cuda.cpp
          MPI_HALOEXCHANGE-OMP.cpp
          MPI_HALOEXCHANGE-OMPTarget.cpp
          MPI_HALOEXCHANGE_FUSED.cpp
          MPI_HALOEXCHANGE_FUSED-Seq.cpp
          MPI_HALOEXCHANGE_FUSED-Hip.cpp
          MPI_HALOEXCHANGE_FUSED-Cuda.cpp
          MPI_HALOEXCHANGE_FUSED-OMP.cpp
          MPI_HALOEXCHANGE_FUSED-OMPTarget.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
