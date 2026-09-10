//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void PRESSURE::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PRESSURE_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto pressure_lam1 = [=](Index_type i) {
                         PRESSURE_BODY1;
                       };
  auto pressure_lam2 = [=](Index_type i) {
                         PRESSURE_BODY2;
                       };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("PRESSURE_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          PRESSURE_BODY1;
        }
        RP_CALI_SUBKERNEL_END("PRESSURE_1");

        RP_CALI_SUBKERNEL_BEGIN("PRESSURE_2");
        for (Index_type i = ibegin; i < iend; ++i ) {
          PRESSURE_BODY2;
        }
        RP_CALI_SUBKERNEL_END("PRESSURE_2");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("PRESSURE_1");
       for (Index_type i = ibegin; i < iend; ++i ) {
         pressure_lam1(i);
       }
       RP_CALI_SUBKERNEL_END("PRESSURE_1");

       RP_CALI_SUBKERNEL_BEGIN("PRESSURE_2");
       for (Index_type i = ibegin; i < iend; ++i ) {
         pressure_lam2(i);
       }
       RP_CALI_SUBKERNEL_END("PRESSURE_2");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RAJA::region<RAJA::seq_region>( [=]() {

          RP_CALI_SUBKERNEL_BEGIN("PRESSURE_1");
          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend), pressure_lam1);
          RP_CALI_SUBKERNEL_END("PRESSURE_1");

          RP_CALI_SUBKERNEL_BEGIN("PRESSURE_2");
          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend), pressure_lam2);
          RP_CALI_SUBKERNEL_END("PRESSURE_2");

        }); // end sequential region (for single-source code)

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  PRESSURE : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(PRESSURE, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace apps
} // end namespace rajaperf
