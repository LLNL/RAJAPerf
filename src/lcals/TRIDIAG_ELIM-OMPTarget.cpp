//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAG_ELIM.hpp"

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


void TRIDIAG_ELIM::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = m_N;

  TRIDIAG_ELIM_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("TRIDIAG_ELIM_1");
      #pragma omp target is_device_ptr(xout, xin, y, z) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        TRIDIAG_ELIM_BODY;
      }
      RP_CALI_SUBKERNEL_END("TRIDIAG_ELIM_1");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("TRIDIAG_ELIM_1");
      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        TRIDIAG_ELIM_BODY;
      });
      RP_CALI_SUBKERNEL_END("TRIDIAG_ELIM_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  TRIDIAG_ELIM : Unknown OMP Tagretvariant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(TRIDIAG_ELIM, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
