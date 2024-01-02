###############################################################################
# Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME comm
  SOURCES HALO_base.cpp
          HALO_PACKING.cpp
          HALO_PACKING-Seq.cpp
          HALO_PACKING-Hip.cpp
          HALO_PACKING-Cuda.cpp
          HALO_PACKING-OMP.cpp
          HALO_PACKING-OMPTarget.cpp
          HALO_PACKING_FUSED.cpp
          HALO_PACKING_FUSED-Seq.cpp
          HALO_PACKING_FUSED-Hip.cpp
          HALO_PACKING_FUSED-Cuda.cpp
          HALO_PACKING_FUSED-OMP.cpp
          HALO_PACKING_FUSED-OMPTarget.cpp
          HALO_SENDRECV.cpp
          HALO_SENDRECV-Seq.cpp
          HALO_SENDRECV-Hip.cpp
          HALO_SENDRECV-Cuda.cpp
          HALO_SENDRECV-OMP.cpp
          HALO_SENDRECV-OMPTarget.cpp
          HALO_EXCHANGE.cpp
          HALO_EXCHANGE-Seq.cpp
          HALO_EXCHANGE-Hip.cpp
          HALO_EXCHANGE-Cuda.cpp
          HALO_EXCHANGE-OMP.cpp
          HALO_EXCHANGE-OMPTarget.cpp
          HALO_EXCHANGE_FUSED.cpp
          HALO_EXCHANGE_FUSED-Seq.cpp
          HALO_EXCHANGE_FUSED-Hip.cpp
          HALO_EXCHANGE_FUSED-Cuda.cpp
          HALO_EXCHANGE_FUSED-OMP.cpp
          HALO_EXCHANGE_FUSED-OMPTarget.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
