###############################################################################
# Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include_directories(.)

# Added as part of the merge with build only infrastructure:
add_subdirectory(common)

if(NOT INFRASTRUCTURE_ONLY)
add_subdirectory(apps)
add_subdirectory(apps-kokkos)
add_subdirectory(basic)
add_subdirectory(basic-kokkos)
#add_subdirectory(kokkos-mechanics)
add_subdirectory(lcals)
add_subdirectory(lcals-kokkos)
#add_subdirectory(polybench)
add_subdirectory(stream)
add_subdirectory(stream-kokkos)
add_subdirectory(algorithm)
add_subdirectory(algorithm-kokkos)
endif()
# Ask David about necessary changes here (wrt to file in Kokkos Kernels)
set(RAJA_PERFSUITE_EXECUTABLE_DEPENDS common)
if(NOT INFRASTRUCTURE_ONLY)
list(APPEND RAJA_PERFSUITE_EXECUTABLE_DEPENDS
    basic
    basic-kokkos
    apps
    apps-kokkos
    #kokkos-mechanics
    lcals
    lcals-kokkos
    #polybench
    stream
    stream-kokkos
    algorithm
    algorithm-kokkos
    )
endif()

# This line must be kept
list(APPEND RAJA_PERFSUITE_EXECUTABLE_DEPENDS ${RAJA_PERFSUITE_DEPENDS}) 

if(ENABLE_TARGET_OPENMP)
  remove_definitions(-DRUN_RAJA_SEQ -DRUN_OPENMP )
include_directories(basic)
include_directories(lcals)
include_directories(apps)
include_directories(algorithm)
include_directories(stream)
include_directories(polybench)
blt_add_executable(
  NAME raja-perf-omptarget.exe
  SOURCES RAJAPerfSuiteDriver.cpp
  apps/AppsData.cpp
  apps/DEL_DOT_VEC_2D.cpp
  apps/DEL_DOT_VEC_2D-Seq.cpp
  apps/DEL_DOT_VEC_2D-OMPTarget.cpp
  apps/DIFFUSION3DPA.cpp
  apps/DIFFUSION3DPA-Seq.cpp
  apps/DIFFUSION3DPA-OMPTarget.cpp
  apps/ENERGY.cpp
  apps/ENERGY-Seq.cpp
  apps/ENERGY-OMPTarget.cpp
  apps/FIR.cpp
  apps/FIR-Seq.cpp
  apps/FIR-OMPTarget.cpp
  apps/PRESSURE.cpp
  apps/PRESSURE-Seq.cpp
  apps/PRESSURE-OMPTarget.cpp
  apps/HALOEXCHANGE.cpp
  apps/HALOEXCHANGE-Seq.cpp
  apps/HALOEXCHANGE-OMPTarget.cpp
  apps/HALOEXCHANGE_FUSED.cpp
  apps/HALOEXCHANGE_FUSED-Seq.cpp
  apps/HALOEXCHANGE_FUSED-OMPTarget.cpp
  apps/LTIMES.cpp
  apps/LTIMES-Seq.cpp
  apps/LTIMES-OMPTarget.cpp
  apps/LTIMES_NOVIEW.cpp
  apps/LTIMES_NOVIEW-Seq.cpp
  apps/LTIMES_NOVIEW-OMPTarget.cpp
  apps/MASS3DPA.cpp
  apps/MASS3DPA-Seq.cpp
  apps/MASS3DPA-OMPTarget.cpp
  apps/VOL3D.cpp
  apps/VOL3D-Seq.cpp
  apps/VOL3D-OMPTarget.cpp
  #apps/WIP-COUPLE.cpp
  #Kokkos bloc
  apps-kokkos/DEL_DOT_VEC_2D-Kokkos.cpp
  apps-kokkos/ENERGY-Kokkos.cpp
  apps-kokkos/FIR-Kokkos.cpp
  apps-kokkos/HALOEXCHANGE-Kokkos.cpp
  apps-kokkos/PRESSURE-Kokkos.cpp
  apps-kokkos/LTIMES-Kokkos.cpp
  apps-kokkos/LTIMES_NOVIEW-Kokkos.cpp
  apps-kokkos/VOL3D-Kokkos.cpp
  #apps-kokkos/WIP-COUPLE.cpp
  #basic/ATOMIC_PI.cpp
  #basic/ATOMIC_PI-Seq.cpp
  #basic/ATOMIC_PI-OMPTarget.cpp
  basic/PI_ATOMIC.cpp
  basic/PI_ATOMIC-Seq.cpp
  basic/PI_ATOMIC-OMPTarget.cpp
  basic/DAXPY.cpp
  basic/DAXPY-Seq.cpp
  basic/DAXPY-OMPTarget.cpp
  basic/IF_QUAD.cpp
  basic/IF_QUAD-Seq.cpp
  basic/IF_QUAD-OMPTarget.cpp
  basic/INIT3.cpp
  basic/INIT3-Seq.cpp
  basic/INIT3-OMPTarget.cpp
  basic/INIT_VIEW1D.cpp
  basic/INIT_VIEW1D-Seq.cpp
  basic/INIT_VIEW1D-OMPTarget.cpp
  basic/INIT_VIEW1D_OFFSET.cpp
  basic/INIT_VIEW1D_OFFSET-Seq.cpp
  basic/INIT_VIEW1D_OFFSET-OMPTarget.cpp
  basic/MAT_MAT_SHARED.cpp
  basic/MAT_MAT_SHARED-Seq.cpp
  basic/MAT_MAT_SHARED-OMPTarget.cpp
  basic/MULADDSUB.cpp
  basic/MULADDSUB-Seq.cpp
  basic/MULADDSUB-OMPTarget.cpp
  basic/NESTED_INIT.cpp
  basic/NESTED_INIT-Seq.cpp
  basic/NESTED_INIT-OMPTarget.cpp
  basic/PI_ATOMIC.cpp
  basic/PI_ATOMIC-Seq.cpp
  basic/PI_ATOMIC-OMPTarget.cpp
  basic/PI_REDUCE.cpp
  basic/PI_REDUCE-Seq.cpp
  basic/PI_REDUCE-OMPTarget.cpp
  basic/REDUCE3_INT.cpp
  basic/REDUCE3_INT-Seq.cpp
  basic/REDUCE3_INT-OMPTarget.cpp
  basic/TRAP_INT.cpp
  basic/TRAP_INT-Seq.cpp
  basic/TRAP_INT-OMPTarget.cpp
  basic-kokkos/PI_ATOMIC-Kokkos.cpp
  basic-kokkos/DAXPY-Kokkos.cpp
  basic-kokkos/IF_QUAD-Kokkos.cpp
  basic-kokkos/INIT3-Kokkos.cpp
  basic-kokkos/INIT_VIEW1D-Kokkos.cpp
  basic-kokkos/INIT_VIEW1D_OFFSET-Kokkos.cpp
  basic-kokkos/MULADDSUB-Kokkos.cpp
  basic-kokkos/NESTED_INIT-Kokkos.cpp
  basic-kokkos/REDUCE3_INT-Kokkos.cpp
  basic-kokkos/TRAP_INT-Kokkos.cpp
  lcals/DIFF_PREDICT.cpp
  lcals/DIFF_PREDICT-Seq.cpp
  lcals/DIFF_PREDICT-OMPTarget.cpp
  lcals/EOS.cpp
  lcals/EOS-Seq.cpp
  lcals/EOS-OMPTarget.cpp
  lcals/FIRST_DIFF.cpp
  lcals/FIRST_DIFF-Seq.cpp
  lcals/FIRST_DIFF-OMPTarget.cpp
  lcals/FIRST_MIN.cpp
  lcals/FIRST_MIN-Seq.cpp
  lcals/FIRST_MIN-OMPTarget.cpp
  lcals/FIRST_SUM.cpp
  lcals/FIRST_SUM-Seq.cpp
  lcals/FIRST_SUM-OMPTarget.cpp
  lcals/GEN_LIN_RECUR.cpp
  lcals/GEN_LIN_RECUR-Seq.cpp
  lcals/GEN_LIN_RECUR-OMPTarget.cpp
  lcals/HYDRO_1D.cpp
  lcals/HYDRO_1D-Seq.cpp
  lcals/HYDRO_1D-OMPTarget.cpp
  lcals/HYDRO_2D.cpp
  lcals/HYDRO_2D-Seq.cpp
  lcals/HYDRO_2D-OMPTarget.cpp
  lcals/INT_PREDICT.cpp
  lcals/INT_PREDICT-Seq.cpp
  lcals/INT_PREDICT-OMPTarget.cpp
  lcals/PLANCKIAN.cpp
  lcals/PLANCKIAN-Seq.cpp
  lcals/PLANCKIAN-OMPTarget.cpp
  lcals/TRIDIAG_ELIM.cpp
  lcals/TRIDIAG_ELIM-Seq.cpp
  lcals/TRIDIAG_ELIM-OMPTarget.cpp
  #Kokkos Bloc
  lcals-kokkos/DIFF_PREDICT-Kokkos.cpp
  lcals-kokkos/EOS-Kokkos.cpp
  lcals-kokkos/FIRST_DIFF-Kokkos.cpp
  lcals-kokkos/FIRST_MIN-Kokkos.cpp
  lcals-kokkos/FIRST_SUM-Kokkos.cpp
  lcals-kokkos/GEN_LIN_RECUR-Kokkos.cpp
  lcals-kokkos/HYDRO_1D-Kokkos.cpp
  lcals-kokkos/HYDRO_2D-Kokkos.cpp
  lcals-kokkos/INT_PREDICT-Kokkos.cpp
  lcals-kokkos/PLANCKIAN-Kokkos.cpp
  lcals-kokkos/TRIDIAG_ELIM-Kokkos.cpp
  #polybench/POLYBENCH_2MM.cpp
  #polybench/POLYBENCH_2MM-Seq.cpp
  #polybench/POLYBENCH_2MM-OMPTarget.cpp
  #polybench/POLYBENCH_3MM.cpp
  #polybench/POLYBENCH_3MM-Seq.cpp
  #polybench/POLYBENCH_3MM-OMPTarget.cpp
  #polybench/POLYBENCH_ADI.cpp
  #polybench/POLYBENCH_ADI-Seq.cpp
  #polybench/POLYBENCH_ADI-OMPTarget.cpp
  #polybench/POLYBENCH_ATAX.cpp
  #polybench/POLYBENCH_ATAX-Seq.cpp
  #polybench/POLYBENCH_ATAX-OMPTarget.cpp
  #polybench/POLYBENCH_FDTD_2D.cpp
  #polybench/POLYBENCH_FDTD_2D-Seq.cpp
  #polybench/POLYBENCH_FDTD_2D-OMPTarget.cpp
  #polybench/POLYBENCH_FLOYD_WARSHALL.cpp
  #polybench/POLYBENCH_FLOYD_WARSHALL-Seq.cpp
  #polybench/POLYBENCH_FLOYD_WARSHALL-OMPTarget.cpp
  #polybench/POLYBENCH_GEMM.cpp
  #polybench/POLYBENCH_GEMM-Seq.cpp
  #polybench/POLYBENCH_GEMM-OMPTarget.cpp
  #polybench/POLYBENCH_GEMVER.cpp
  #polybench/POLYBENCH_GEMVER-Seq.cpp
  #polybench/POLYBENCH_GEMVER-OMPTarget.cpp
  #polybench/POLYBENCH_GESUMMV.cpp
  #polybench/POLYBENCH_GESUMMV-Seq.cpp
  #polybench/POLYBENCH_GESUMMV-OMPTarget.cpp
  #polybench/POLYBENCH_HEAT_3D.cpp
  #polybench/POLYBENCH_HEAT_3D-Seq.cpp
  #polybench/POLYBENCH_HEAT_3D-OMPTarget.cpp
  #polybench/POLYBENCH_JACOBI_1D.cpp
  #polybench/POLYBENCH_JACOBI_1D-Seq.cpp
  #polybench/POLYBENCH_JACOBI_1D-OMPTarget.cpp
  #polybench/POLYBENCH_JACOBI_2D.cpp
  #polybench/POLYBENCH_JACOBI_2D-Seq.cpp
  #polybench/POLYBENCH_JACOBI_2D-OMPTarget.cpp
  #polybench/POLYBENCH_MVT.cpp
  #polybench/POLYBENCH_MVT-Seq.cpp
  #polybench/POLYBENCH_MVT-OMPTarget.cpp
  stream/ADD.cpp
  stream/ADD-Seq.cpp
  stream/ADD-OMPTarget.cpp
  stream/COPY.cpp
  stream/COPY-Seq.cpp
  stream/COPY-OMPTarget.cpp
  stream/DOT.cpp
  stream/DOT-Seq.cpp
  stream/DOT-OMPTarget.cpp
  stream/MUL.cpp
  stream/MUL-Seq.cpp
  stream/MUL-OMPTarget.cpp
  stream/TRIAD.cpp
  stream/TRIAD-Seq.cpp
  stream/TRIAD-OMPTarget.cpp
  stream-kokkos/ADD-Kokkos.cpp
  stream-kokkos/COPY-Kokkos.cpp
  stream-kokkos/DOT-Kokkos.cpp
  stream-kokkos/MUL-Kokkos.cpp
  stream-kokkos/TRIAD-Kokkos.cpp
  algorithm/SORT.cpp
  algorithm/SORT-Seq.cpp
  algorithm/SORTPAIRS.cpp
  algorithm/SORTPAIRS-Seq.cpp
  algorithm-kokkos/SORT-Kokkos.cpp
  algorithm-kokkos/SORTPAIRS-Kokkos.cpp
  common/DataUtils.cpp
  common/Executor.cpp
  common/KernelBase.cpp
  common/OutputUtils.cpp
  common/RAJAPerfSuite.cpp
  common/RPTypes.hpp
  common/RunParams.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_DEPENDS}
)

else()
	if(NOT INFRASTRUCTURE_ONLY)
blt_add_executable(
  NAME raja-perf.exe
  SOURCES RAJAPerfSuiteDriver.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_EXECUTABLE_DEPENDS}
  )
	endif()
endif()
