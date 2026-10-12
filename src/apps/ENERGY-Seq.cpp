//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void ENERGY::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ENERGY_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto energy_lam1 = [=](Index_type i) {
                       ENERGY_BODY1;
                     };
  auto energy_lam2 = [=](Index_type i) {
                       ENERGY_BODY2;
                     };
  auto energy_lam3 = [=](Index_type i) {
                       ENERGY_BODY3;
                     };
  auto energy_lam4 = [=](Index_type i) {
                       ENERGY_BODY4;
                     };
  auto energy_lam5 = [=](Index_type i) {
                       ENERGY_BODY5;
                     };
  auto energy_lam6 = [=](Index_type i) {
                       ENERGY_BODY6;
                     };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY1;
        }
        RP_CALI_SUBKERNEL_END("ENERGY_1");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_2");
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY2;
        }
        RP_CALI_SUBKERNEL_END("ENERGY_2");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_3");
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY3;
        }
        RP_CALI_SUBKERNEL_END("ENERGY_3");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_4");
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY4;
        }
        RP_CALI_SUBKERNEL_END("ENERGY_4");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_5");
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY5;
        }
        RP_CALI_SUBKERNEL_END("ENERGY_5");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_6");
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY6;
        }
        RP_CALI_SUBKERNEL_END("ENERGY_6");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          energy_lam1(i);
        }
        RP_CALI_SUBKERNEL_END("ENERGY_1");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_2");
        for (Index_type i = ibegin; i < iend; ++i ) {
          energy_lam2(i);
        }
        RP_CALI_SUBKERNEL_END("ENERGY_2");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_3");
        for (Index_type i = ibegin; i < iend; ++i ) {
          energy_lam3(i);
        }
        RP_CALI_SUBKERNEL_END("ENERGY_3");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_4");
        for (Index_type i = ibegin; i < iend; ++i ) {
          energy_lam4(i);
        }
        RP_CALI_SUBKERNEL_END("ENERGY_4");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_5");
        for (Index_type i = ibegin; i < iend; ++i ) {
          energy_lam5(i);
        }
        RP_CALI_SUBKERNEL_END("ENERGY_5");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_6");
        for (Index_type i = ibegin; i < iend; ++i ) {
          energy_lam6(i);
        }
        RP_CALI_SUBKERNEL_END("ENERGY_6");

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

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_1");
          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend), energy_lam1);
          RP_CALI_SUBKERNEL_END("ENERGY_1");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_2");
          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend), energy_lam2);
          RP_CALI_SUBKERNEL_END("ENERGY_2");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_3");
          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend), energy_lam3);
          RP_CALI_SUBKERNEL_END("ENERGY_3");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_4");
          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend), energy_lam4);
          RP_CALI_SUBKERNEL_END("ENERGY_4");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_5");
          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend), energy_lam5);
          RP_CALI_SUBKERNEL_END("ENERGY_5");

          RP_CALI_SUBKERNEL_BEGIN("ENERGY_6");
          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend), energy_lam6);
          RP_CALI_SUBKERNEL_END("ENERGY_6");

        }); // end sequential region (for single-source code)

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  ENERGY : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(ENERGY, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace apps
} // end namespace rajaperf
