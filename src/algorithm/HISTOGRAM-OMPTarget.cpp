//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HISTOGRAM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;


void HISTOGRAM::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  HISTOGRAM_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    HISTOGRAM_SETUP_COUNTS;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("HISTOGRAM_1");
      initOpenMPDeviceData(counts, counts_init, num_bins);

      #pragma omp target is_device_ptr(counts, bins)
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        HISTOGRAM_BODY(RAJAPERF_ATOMIC_ADD_OMP);
      }

      getOpenMPDeviceData(counts_final, counts, num_bins);
      RP_CALI_SUBKERNEL_END("HISTOGRAM_1");

    }
    stopTimer();

    HISTOGRAM_TEARDOWN_COUNTS;

  } else {
     getCout() << "\n  HISTOGRAM : Unknown OMP Target variant id = " << vid << std::endl;
  }

  HISTOGRAM_DATA_TEARDOWN;

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(HISTOGRAM, OpenMPTarget, Base_OpenMPTarget)

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
