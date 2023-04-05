//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_1D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

void POLYBENCH_JACOBI_1D::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_JACOBI_1D_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        #pragma omp target is_device_ptr(A,B) device( did )
        #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
        for (Index_type i = 1; i < N-1; ++i ) {
          POLYBENCH_JACOBI_1D_BODY1;
        }

        #pragma omp target is_device_ptr(A,B) device( did )
        #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
        for (Index_type i = 1; i < N-1; ++i ) {
          POLYBENCH_JACOBI_1D_BODY2;
        }
      }

    }
    stopTimer();

  } else if (vid == RAJA_OpenMPTarget ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>> (
          RAJA::RangeSegment{1, N-1}, [=] (Index_type i) {
            POLYBENCH_JACOBI_1D_BODY1;
        });

        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>> (
          RAJA::RangeSegment{1, N-1}, [=] (Index_type i) {
            POLYBENCH_JACOBI_1D_BODY2;
        });

      }

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_JACOBI_1D : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

