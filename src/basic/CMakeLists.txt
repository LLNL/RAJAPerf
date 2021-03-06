###############################################################################
# Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME basic
  SOURCES DAXPY.cpp
          DAXPY-Seq.cpp
          DAXPY-Hip.cpp
          DAXPY-Cuda.cpp
          DAXPY-OMP.cpp
          DAXPY-OMPTarget.cpp
          IF_QUAD.cpp
          IF_QUAD-Seq.cpp
          IF_QUAD-Hip.cpp
          IF_QUAD-Cuda.cpp
          IF_QUAD-OMP.cpp
          IF_QUAD-OMPTarget.cpp
          INIT3.cpp
          INIT3-Seq.cpp
          INIT3-Hip.cpp
          INIT3-Cuda.cpp
          INIT3-OMP.cpp
          INIT3-OMPTarget.cpp
          INIT_VIEW1D.cpp
          INIT_VIEW1D-Seq.cpp
          INIT_VIEW1D-Hip.cpp
          INIT_VIEW1D-Cuda.cpp
          INIT_VIEW1D-OMP.cpp
          INIT_VIEW1D-OMPTarget.cpp
          INIT_VIEW1D_OFFSET.cpp
          INIT_VIEW1D_OFFSET-Seq.cpp
          INIT_VIEW1D_OFFSET-Hip.cpp
          INIT_VIEW1D_OFFSET-Cuda.cpp
          INIT_VIEW1D_OFFSET-OMP.cpp
          INIT_VIEW1D_OFFSET-OMPTarget.cpp
          MULADDSUB.cpp
          MULADDSUB-Seq.cpp
          MULADDSUB-Hip.cpp
          MULADDSUB-Cuda.cpp
          MULADDSUB-OMP.cpp
          MULADDSUB-OMPTarget.cpp
          NESTED_INIT.cpp
          NESTED_INIT-Seq.cpp
          NESTED_INIT-Hip.cpp
          NESTED_INIT-Cuda.cpp
          NESTED_INIT-OMP.cpp
          NESTED_INIT-OMPTarget.cpp
          PI_ATOMIC.cpp
          PI_ATOMIC-Seq.cpp
          PI_ATOMIC-Hip.cpp
          PI_ATOMIC-Cuda.cpp
          PI_ATOMIC-OMP.cpp
          PI_ATOMIC-OMPTarget.cpp
          PI_REDUCE.cpp
          PI_REDUCE-Seq.cpp
          PI_REDUCE-Hip.cpp
          PI_REDUCE-Cuda.cpp
          PI_REDUCE-OMP.cpp
          PI_REDUCE-OMPTarget.cpp
          REDUCE3_INT.cpp
          REDUCE3_INT-Seq.cpp
          REDUCE3_INT-Hip.cpp
          REDUCE3_INT-Cuda.cpp
          REDUCE3_INT-OMP.cpp
          REDUCE3_INT-OMPTarget.cpp
          TRAP_INT.cpp
          TRAP_INT-Seq.cpp
          TRAP_INT-Hip.cpp
          TRAP_INT-Cuda.cpp
          TRAP_INT-OMPTarget.cpp
          TRAP_INT-OMP.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
