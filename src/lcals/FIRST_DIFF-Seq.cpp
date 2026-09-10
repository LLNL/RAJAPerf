//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_DIFF.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{


void FIRST_DIFF::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  FIRST_DIFF_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto firstdiff_lam = [=](Index_type i) {
                         FIRST_DIFF_BODY;
                       };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("FIRST_DIFF_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          FIRST_DIFF_BODY;
        }
        RP_CALI_SUBKERNEL_END("FIRST_DIFF_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("FIRST_DIFF_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          firstdiff_lam(i);
        }
        RP_CALI_SUBKERNEL_END("FIRST_DIFF_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("FIRST_DIFF_1");
        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend), firstdiff_lam);
        RP_CALI_SUBKERNEL_END("FIRST_DIFF_1");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  FIRST_DIFF : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(FIRST_DIFF, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace lcals
} // end namespace rajaperf
