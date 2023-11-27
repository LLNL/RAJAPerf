###############################################################################
# Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME comm
  SOURCES HALO_base.cpp
          HALOPACKING.cpp
          HALOPACKING-Seq.cpp
          HALOPACKING-Hip.cpp
          HALOPACKING-Cuda.cpp
          HALOPACKING-OMP.cpp
          HALOPACKING-OMPTarget.cpp
          HALOPACKING_FUSED.cpp
          HALOPACKING_FUSED-Seq.cpp
          HALOPACKING_FUSED-Hip.cpp
          HALOPACKING_FUSED-Cuda.cpp
          HALOPACKING_FUSED-OMP.cpp
          HALOPACKING_FUSED-OMPTarget.cpp
          MPI_HALOSENDRECV.cpp
          MPI_HALOSENDRECV-Seq.cpp
          MPI_HALOSENDRECV-Hip.cpp
          MPI_HALOSENDRECV-Cuda.cpp
          MPI_HALOSENDRECV-OMP.cpp
          MPI_HALOSENDRECV-OMPTarget.cpp
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
