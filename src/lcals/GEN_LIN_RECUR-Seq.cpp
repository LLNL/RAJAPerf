//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{


void GEN_LIN_RECUR::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  GEN_LIN_RECUR_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto genlinrecur_lam1 = [=](Index_type k) {
                            GEN_LIN_RECUR_BODY1;
                          };
  auto genlinrecur_lam2 = [=](Index_type i) {
                            GEN_LIN_RECUR_BODY2;
                          };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_1");
        for (Index_type k = 0; k < N; ++k ) {
          GEN_LIN_RECUR_BODY1;
        }
        RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_1");

        RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_2");
        for (Index_type i = 1; i < N+1; ++i ) {
          GEN_LIN_RECUR_BODY2;
        }
        RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_2");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_1");
        for (Index_type k = 0; k < N; ++k ) {
          genlinrecur_lam1(k);
        }
        RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_1");

        RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_2");
        for (Index_type i = 1; i < N+1; ++i ) {
          genlinrecur_lam2(i);
        }
        RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_2");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_1");
        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(0, N), genlinrecur_lam1);
        RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_1");

        RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_2");
        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(1, N+1), genlinrecur_lam2);
        RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_2");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  GEN_LIN_RECUR : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(GEN_LIN_RECUR, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace lcals
} // end namespace rajaperf
