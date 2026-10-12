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

#include <iostream>

namespace rajaperf
{
namespace basic
{


void MULTI_REDUCE::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MULTI_REDUCE_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      MULTI_REDUCE_SETUP_VALUES;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MULTI_REDUCE_1");
        MULTI_REDUCE_INIT_VALUES;

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          MULTI_REDUCE_BODY(RAJAPERF_ATOMIC_ADD_OMP);
        }

        MULTI_REDUCE_FINALIZE_VALUES;
        RP_CALI_SUBKERNEL_END("MULTI_REDUCE_1");

      }
      stopTimer();

      MULTI_REDUCE_TEARDOWN_VALUES;

      break;
    }

    case Lambda_OpenMP : {

      MULTI_REDUCE_SETUP_VALUES;

      auto multi_reduce_base_lam = [=](Index_type i) {
            MULTI_REDUCE_BODY(RAJAPERF_ATOMIC_ADD_OMP);
          };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MULTI_REDUCE_1");
        MULTI_REDUCE_INIT_VALUES;

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          multi_reduce_base_lam(i);
        }

        MULTI_REDUCE_FINALIZE_VALUES;
        RP_CALI_SUBKERNEL_END("MULTI_REDUCE_1");

      }
      stopTimer();

      MULTI_REDUCE_TEARDOWN_VALUES;

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MULTI_REDUCE_1");
        MULTI_REDUCE_INIT_VALUES_RAJA(RAJA::omp_multi_reduce);

        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            MULTI_REDUCE_BODY(RAJAPERF_ADD);
        });

        MULTI_REDUCE_FINALIZE_VALUES_RAJA(RAJA::omp_multi_reduce);
        RP_CALI_SUBKERNEL_END("MULTI_REDUCE_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  MULTI_REDUCE : Unknown variant id = " << vid << std::endl;
    }

  }

  MULTI_REDUCE_DATA_TEARDOWN;

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(MULTI_REDUCE, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace basic
} // end namespace rajaperf
