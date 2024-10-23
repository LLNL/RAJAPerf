//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HISTOGRAM.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void HISTOGRAM::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  HISTOGRAM_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      HISTOGRAM_SETUP_COUNTS;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        HISTOGRAM_INIT_COUNTS;

        for (Index_type i = ibegin; i < iend; ++i ) {
          HISTOGRAM_BODY;
        }

        HISTOGRAM_FINALIZE_COUNTS;

      }
      stopTimer();

      HISTOGRAM_TEARDOWN_COUNTS;

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      HISTOGRAM_SETUP_COUNTS;

      auto histogram_base_lam = [=](Index_type i) {
                                 HISTOGRAM_BODY;
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        HISTOGRAM_INIT_COUNTS;

        for (Index_type i = ibegin; i < iend; ++i ) {
          histogram_base_lam(i);
        }

        HISTOGRAM_FINALIZE_COUNTS;

      }
      stopTimer();

      HISTOGRAM_TEARDOWN_COUNTS;

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        HISTOGRAM_INIT_COUNTS_RAJA(RAJA::seq_multi_reduce);

        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
            HISTOGRAM_BODY;
        });

        HISTOGRAM_FINALIZE_COUNTS_RAJA(RAJA::seq_multi_reduce);

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  HISTOGRAM : Unknown variant id = " << vid << std::endl;
    }

  }

  HISTOGRAM_DATA_TEARDOWN;

}

} // end namespace algorithm
} // end namespace rajaperf
