//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void NESTED_INIT::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto nestedinit_lam = [=](Index_type i, Index_type j, Index_type k) {
                          NESTED_INIT_BODY;
                        };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
        for (Index_type k = 0; k < nk; ++k ) {
          for (Index_type j = 0; j < nj; ++j ) {
            for (Index_type i = 0; i < ni; ++i ) {
              NESTED_INIT_BODY;
            }
          }
        }
        RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
          for (Index_type k = 0; k < nk; ++k ) {
            for (Index_type j = 0; j < nj; ++j ) {
              for (Index_type i = 0; i < ni; ++i ) {
                nestedinit_lam(i, j, k);
              }
            }
          }
          RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<2, RAJA::seq_exec,    // k
            RAJA::statement::For<1, RAJA::seq_exec,  // j
              RAJA::statement::For<0, RAJA::seq_exec,// i
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
        RAJA::kernel_resource<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                                          RAJA::RangeSegment(0, nj),
                                                          RAJA::RangeSegment(0, nk)),
                                         res,
                                         nestedinit_lam
                                       );
        RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(NESTED_INIT, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace basic
} // end namespace rajaperf
