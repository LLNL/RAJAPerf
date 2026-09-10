//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D_OFFSET.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;


void INIT_VIEW1D_OFFSET::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getActualProblemSize()+1;

  INIT_VIEW1D_OFFSET_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("INIT_VIEW1D_OFFSET_1");
      #pragma omp target is_device_ptr(a) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        INIT_VIEW1D_OFFSET_BODY;
      }
      RP_CALI_SUBKERNEL_END("INIT_VIEW1D_OFFSET_1");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

     auto res{getOmpTargetResource()};

     INIT_VIEW1D_OFFSET_VIEW_RAJA;

     startTimer();
     // Loop counter increment uses macro to quiet C++20 compiler warning
     for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("INIT_VIEW1D_OFFSET_1");
       RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
         RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
         INIT_VIEW1D_OFFSET_BODY_RAJA;
       });
       RP_CALI_SUBKERNEL_END("INIT_VIEW1D_OFFSET_1");

     }
     stopTimer();

  } else {
     getCout() << "\n  INIT_VIEW1D_OFFSET : Unknown OMP Targetvariant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(INIT_VIEW1D_OFFSET, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

