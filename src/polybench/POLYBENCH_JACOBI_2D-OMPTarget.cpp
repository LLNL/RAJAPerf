//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

void POLYBENCH_JACOBI_2D::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        #pragma omp target is_device_ptr(A,B) device( did )
        #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
        for (Index_type i = 1; i < N-1; ++i ) {
          for (Index_type j = 1; j < N-1; ++j ) {
            POLYBENCH_JACOBI_2D_BODY1;
          }
        }

        #pragma omp target is_device_ptr(A,B) device( did )
        #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
        for (Index_type i = 1; i < N-1; ++i ) {
          for (Index_type j = 1; j < N-1; ++j ) {
            POLYBENCH_JACOBI_2D_BODY2;
          }
        }

      }

    }
    stopTimer();

  } else if (vid == RAJA_OpenMPTarget) {

    POLYBENCH_JACOBI_2D_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>,
          RAJA::statement::Lambda<0>
        >,
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>,
          RAJA::statement::Lambda<1>
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                 RAJA::RangeSegment{1, N-1}),
          [=] (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY1_RAJA;
          },
          [=] (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY2_RAJA;
          }
        );

      }

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_JACOBI_2D : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

