//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SCAN.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>
#include <vector>

namespace rajaperf
{
namespace algorithm
{

  //
  // Define threads per team for target execution
  //
#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
  const size_t threads_per_team = 256;
#endif


void SCAN::runOpenMPTargetVariant(VariantID vid)
{
#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SCAN_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMPTarget : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("SCAN_1");
        SCAN_PROLOGUE;

        #pragma omp target is_device_ptr(x,y) device( did )
        #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) \
                                                  reduction(inscan, +:scan_var)
        for (Index_type i = ibegin; i < iend; ++i ) {
          y[i] = scan_var;
          #pragma omp scan exclusive(scan_var)
          scan_var += x[i];
        }
        RP_CALI_SUBKERNEL_END("SCAN_1");

      }
      stopTimer();

      break;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(SCAN, OpenMPTarget, Base_OpenMPTarget)
#else
void SCAN::defineOpenMPTargetVariantTunings() {}
#endif

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
