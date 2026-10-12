//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;


void PRESSURE::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PRESSURE_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("PRESSURE_1");
      #pragma omp target is_device_ptr(compression, bvc) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        PRESSURE_BODY1;
      }
      RP_CALI_SUBKERNEL_END("PRESSURE_1");

      RP_CALI_SUBKERNEL_BEGIN("PRESSURE_2");
      #pragma omp target is_device_ptr(bvc, p_new, e_old, vnewc) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        PRESSURE_BODY2;
      }
      RP_CALI_SUBKERNEL_END("PRESSURE_2");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RP_CALI_SUBKERNEL_BEGIN("PRESSURE_1");
        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          PRESSURE_BODY1;
        });
        RP_CALI_SUBKERNEL_END("PRESSURE_1");

        RP_CALI_SUBKERNEL_BEGIN("PRESSURE_2");
        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          PRESSURE_BODY2;
        });
        RP_CALI_SUBKERNEL_END("PRESSURE_2");

      }); // end sequential region (for single-source code)

    }
    stopTimer();

  } else {
    getCout() << "\n  PRESSURE : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(PRESSURE, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
