//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MULTI_REDUCE.hpp"

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


void MULTI_REDUCE::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MULTI_REDUCE_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    MULTI_REDUCE_SETUP_VALUES;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MULTI_REDUCE_1");
      initOpenMPDeviceData(values, values_init, num_bins);

      #pragma omp target is_device_ptr(values, bins, data)
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        MULTI_REDUCE_BODY(RAJAPERF_ATOMIC_ADD_OMP);
      }

      getOpenMPDeviceData(values_final, values, num_bins);
      RP_CALI_SUBKERNEL_END("MULTI_REDUCE_1");

    }
    stopTimer();

    MULTI_REDUCE_TEARDOWN_VALUES;

  } else {
     getCout() << "\n  MULTI_REDUCE : Unknown OMP Target variant id = " << vid << std::endl;
  }

  MULTI_REDUCE_DATA_TEARDOWN;

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(MULTI_REDUCE, OpenMPTarget, Base_OpenMPTarget)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
