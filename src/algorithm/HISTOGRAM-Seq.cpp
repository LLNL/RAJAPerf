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

#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void HISTOGRAM::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  HISTOGRAM_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      HISTOGRAM_SETUP_COUNTS;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("HISTOGRAM_1");
        HISTOGRAM_INIT_COUNTS;

        for (Index_type i = ibegin; i < iend; ++i ) {
          HISTOGRAM_BODY(RAJAPERF_ATOMIC_ADD_SEQ);
        }

        HISTOGRAM_FINALIZE_COUNTS;
        RP_CALI_SUBKERNEL_END("HISTOGRAM_1");

      }
      stopTimer();

      HISTOGRAM_TEARDOWN_COUNTS;

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      HISTOGRAM_SETUP_COUNTS;

      auto histogram_base_lam = [=](Index_type i) {
                                  HISTOGRAM_BODY(RAJAPERF_ATOMIC_ADD_SEQ);
                               };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("HISTOGRAM_1");
        HISTOGRAM_INIT_COUNTS;

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

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("HISTOGRAM_1");
        HISTOGRAM_INIT_COUNTS_RAJA(RAJA::seq_multi_reduce);

        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
            HISTOGRAM_BODY(RAJAPERF_ADD);
        });

        HISTOGRAM_FINALIZE_COUNTS_RAJA(RAJA::seq_multi_reduce);
        RP_CALI_SUBKERNEL_END("HISTOGRAM_1");

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  HISTOGRAM : Unknown variant id = " << vid << std::endl;
    }

  }

  HISTOGRAM_DATA_TEARDOWN;

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(HISTOGRAM, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace algorithm
} // end namespace rajaperf
