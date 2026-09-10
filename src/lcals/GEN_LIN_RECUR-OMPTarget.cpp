//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;


void GEN_LIN_RECUR::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  GEN_LIN_RECUR_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_1");
      #pragma omp target is_device_ptr(b5, stb5, sa, sb) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type k = 0; k < N; ++k ) {
        GEN_LIN_RECUR_BODY1;
      }
      RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_1");

      RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_2");
      #pragma omp target is_device_ptr(b5, stb5, sa, sb) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = 1; i < N+1; ++i ) {
        GEN_LIN_RECUR_BODY2;
      }
      RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_2");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_1");
      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
        RAJA::RangeSegment(0, N), [=] (Index_type k) {
        GEN_LIN_RECUR_BODY1;
      });
      RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_1");

      RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_2");
      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
        RAJA::RangeSegment(1, N+1), [=] (Index_type i) {
        GEN_LIN_RECUR_BODY2;
      });
      RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_2");

    }
    stopTimer();

  } else {
     getCout() << "\n  GEN_LIN_RECUR : Unknown OMP Tagretvariant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(GEN_LIN_RECUR, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
