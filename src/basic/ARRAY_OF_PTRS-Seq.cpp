//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ARRAY_OF_PTRS.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void ARRAY_OF_PTRS::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ARRAY_OF_PTRS_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto array_of_ptrs_lam = [=](Index_type i) {
                     ARRAY_OF_PTRS_BODY(x);
                   };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("ARRAY_OF_PTRS_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          ARRAY_OF_PTRS_BODY(x);
        }
        RP_CALI_SUBKERNEL_END("ARRAY_OF_PTRS_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("ARRAY_OF_PTRS_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          array_of_ptrs_lam(i);
        }
        RP_CALI_SUBKERNEL_END("ARRAY_OF_PTRS_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("ARRAY_OF_PTRS_1");
        RAJA::forall<RAJA::simd_exec>( res,
          RAJA::RangeSegment(ibegin, iend), array_of_ptrs_lam);
        RP_CALI_SUBKERNEL_END("ARRAY_OF_PTRS_1");

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  ARRAY_OF_PTRS : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(ARRAY_OF_PTRS, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace basic
} // end namespace rajaperf
