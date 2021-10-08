###############################################################################
# Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME apps
  SOURCES AppsData.cpp
          DEL_DOT_VEC_2D.cpp 
          DEL_DOT_VEC_2D-Seq.cpp 
          DEL_DOT_VEC_2D-Hip.cpp 
          DEL_DOT_VEC_2D-Cuda.cpp 
          DEL_DOT_VEC_2D-Sycl.cpp 
          DEL_DOT_VEC_2D-OMP.cpp 
          DEL_DOT_VEC_2D-OMPTarget.cpp 
          DIFFUSION3DPA.cpp
          DIFFUSION3DPA-Cuda.cpp
          DIFFUSION3DPA-Hip.cpp
          DIFFUSION3DPA-Seq.cpp
          DIFFUSION3DPA-OMP.cpp
          DIFFUSION3DPA-OMPTarget.cpp
          ENERGY.cpp
          ENERGY-Seq.cpp
          ENERGY-Hip.cpp 
          ENERGY-Cuda.cpp 
          ENERGY-Sycl.cpp 
          ENERGY-OMP.cpp 
          ENERGY-OMPTarget.cpp 
          FIR.cpp
          FIR-Seq.cpp
          FIR-Hip.cpp
          FIR-Cuda.cpp
          FIR-Sycl.cpp
          FIR-OMP.cpp
          FIR-OMPTarget.cpp
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
          LTIMES.cpp
          LTIMES-Seq.cpp
          LTIMES-Hip.cpp
          LTIMES-Cuda.cpp
          LTIMES-Sycl.cpp
          LTIMES-OMP.cpp
          LTIMES-OMPTarget.cpp
          LTIMES_NOVIEW.cpp
          LTIMES_NOVIEW-Seq.cpp
          LTIMES_NOVIEW-Hip.cpp
          LTIMES_NOVIEW-Cuda.cpp
          LTIMES_NOVIEW-Sycl.cpp
          LTIMES_NOVIEW-OMP.cpp
          LTIMES_NOVIEW-OMPTarget.cpp
          MASS3DPA.cpp
          MASS3DPA-Cuda.cpp
          MASS3DPA-Hip.cpp
          MASS3DPA-Seq.cpp
          MASS3DPA-OMP.cpp
          MASS3DPA-OMPTarget.cpp
          PRESSURE.cpp 
          PRESSURE-Seq.cpp 
          PRESSURE-Hip.cpp 
          PRESSURE-Cuda.cpp 
          PRESSURE-Sycl.cpp 
          PRESSURE-OMP.cpp 
          PRESSURE-OMPTarget.cpp 
          VOL3D.cpp
          VOL3D-Seq.cpp
          VOL3D-Hip.cpp 
          VOL3D-Cuda.cpp 
          VOL3D-Sycl.cpp 
          VOL3D-OMP.cpp 
          VOL3D-OMPTarget.cpp 
          WIP-COUPLE.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
