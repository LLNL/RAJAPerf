//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SCAN.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void SCAN::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SCAN_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("SCAN_1");
        SCAN_PROLOGUE;
        for (Index_type i = ibegin; i < iend; ++i ) {
          SCAN_BODY;
        }
        RP_CALI_SUBKERNEL_END("SCAN_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("SCAN_1");
        SCAN_PROLOGUE;
        auto scan_lam = [=, &scan_var](Index_type i) {
                          SCAN_BODY;
                        };
        for (Index_type i = ibegin; i < iend; ++i ) {
          scan_lam(i);
        }
        RP_CALI_SUBKERNEL_END("SCAN_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("SCAN_1");
        RAJA::exclusive_scan<RAJA::seq_exec>(res, RAJA_SCAN_ARGS);
        RP_CALI_SUBKERNEL_END("SCAN_1");

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  SCAN : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(SCAN, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace algorithm
} // end namespace rajaperf
