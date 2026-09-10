//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf
{
namespace apps
{


void FIR::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  FIR_COEFF;

  FIR_DATA_SETUP;

  Real_type coeff[FIR_COEFFLEN];
  std::copy(std::begin(coeff_array), std::end(coeff_array), std::begin(coeff));

#if defined(RUN_RAJA_SEQ)
  auto fir_lam = [=](Index_type i) {
                   FIR_BODY;
                 };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("FIR_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          FIR_BODY;
        }
        RP_CALI_SUBKERNEL_END("FIR_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("FIR_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
           fir_lam(i);
        }
        RP_CALI_SUBKERNEL_END("FIR_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()}; 

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("FIR_1");
        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend), fir_lam);
        RP_CALI_SUBKERNEL_END("FIR_1");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  FIR : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(FIR, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace apps
} // end namespace rajaperf
