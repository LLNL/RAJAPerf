//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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

void POLYBENCH_JACOBI_1D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_JACOBI_1D_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_1");
      #pragma omp target is_device_ptr(A,B) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = 1; i < N-1; ++i ) {
        POLYBENCH_JACOBI_1D_BODY1;
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_2");
      #pragma omp target is_device_ptr(A,B) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = 1; i < N-1; ++i ) {
        POLYBENCH_JACOBI_1D_BODY2;
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_2");

    }
    stopTimer();

  } else if (vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_1");
      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>> ( res,
        RAJA::RangeSegment{1, N-1}, [=] (Index_type i) {
          POLYBENCH_JACOBI_1D_BODY1;
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_2");
      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>> ( res,
        RAJA::RangeSegment{1, N-1}, [=] (Index_type i) {
          POLYBENCH_JACOBI_1D_BODY2;
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_2");

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_JACOBI_1D : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_JACOBI_1D, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
