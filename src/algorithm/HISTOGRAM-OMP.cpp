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

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

void HISTOGRAM::runOpenMPVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  HISTOGRAM_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      HISTOGRAM_SETUP_COUNTS;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("HISTOGRAM_1");
        HISTOGRAM_INIT_COUNTS;

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          HISTOGRAM_BODY(RAJAPERF_ATOMIC_ADD_OMP);
        }

        HISTOGRAM_FINALIZE_COUNTS;
        RP_CALI_SUBKERNEL_END("HISTOGRAM_1");

      }
      stopTimer();

      HISTOGRAM_TEARDOWN_COUNTS;

      break;
    }

    case Lambda_OpenMP : {

      HISTOGRAM_SETUP_COUNTS;

      auto histogram_base_lam = [=](Index_type i) {
            HISTOGRAM_BODY(RAJAPERF_ATOMIC_ADD_OMP);
          };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("HISTOGRAM_1");
        HISTOGRAM_INIT_COUNTS;

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          histogram_base_lam(i);
        }

        HISTOGRAM_FINALIZE_COUNTS;
        RP_CALI_SUBKERNEL_END("HISTOGRAM_1");

      }
      stopTimer();

      HISTOGRAM_TEARDOWN_COUNTS;

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("HISTOGRAM_1");
        HISTOGRAM_INIT_COUNTS_RAJA(RAJA::omp_multi_reduce);

        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            HISTOGRAM_BODY(RAJAPERF_ADD);
        });

        HISTOGRAM_FINALIZE_COUNTS_RAJA(RAJA::omp_multi_reduce);
        RP_CALI_SUBKERNEL_END("HISTOGRAM_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  HISTOGRAM : Unknown variant id = " << vid << std::endl;
    }

  }

  HISTOGRAM_DATA_TEARDOWN;

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(HISTOGRAM, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_OPENMP
