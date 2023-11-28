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
          HALOSENDRECV.cpp
          HALOSENDRECV-Seq.cpp
          HALOSENDRECV-Hip.cpp
          HALOSENDRECV-Cuda.cpp
          HALOSENDRECV-OMP.cpp
          HALOSENDRECV-OMPTarget.cpp
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
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
