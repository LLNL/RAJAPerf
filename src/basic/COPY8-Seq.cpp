//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "COPY8.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void COPY8::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  COPY8_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto copy8_lam = [=](Index_type i) {
                     COPY8_BODY;
                   };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("COPY8_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          COPY8_BODY;
        }
        RP_CALI_SUBKERNEL_END("COPY8_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("COPY8_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          copy8_lam(i);
        }
        RP_CALI_SUBKERNEL_END("COPY8_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("COPY8_1");
        RAJA::forall<RAJA::simd_exec>( res,
          RAJA::RangeSegment(ibegin, iend), copy8_lam);
        RP_CALI_SUBKERNEL_END("COPY8_1");

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  COPY8 : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(COPY8, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace basic
} // end namespace rajaperf
