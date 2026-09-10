//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PLANCKIAN.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>
#include <cmath>

namespace rajaperf
{
namespace lcals
{


void PLANCKIAN::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PLANCKIAN_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto planckian_lam = [=](Index_type i) {
                         PLANCKIAN_BODY;
                       };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("PLANCKIAN_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          PLANCKIAN_BODY;
        }
        RP_CALI_SUBKERNEL_END("PLANCKIAN_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("PLANCKIAN_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          planckian_lam(i);
        }
        RP_CALI_SUBKERNEL_END("PLANCKIAN_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("PLANCKIAN_1");
        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend), planckian_lam);
        RP_CALI_SUBKERNEL_END("PLANCKIAN_1");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  PLANCKIAN : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(PLANCKIAN, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace lcals
} // end namespace rajaperf
