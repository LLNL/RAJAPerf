//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MULTI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void MULTI_REDUCE::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MULTI_REDUCE_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      MULTI_REDUCE_SETUP_VALUES;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        MULTI_REDUCE_INIT_VALUES;

        for (Index_type i = ibegin; i < iend; ++i ) {
          MULTI_REDUCE_BODY;
        }

        MULTI_REDUCE_FINALIZE_VALUES;

      }
      stopTimer();

      MULTI_REDUCE_TEARDOWN_VALUES;

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      MULTI_REDUCE_SETUP_VALUES;

      auto multi_reduce_base_lam = [=](Index_type i) {
                                 MULTI_REDUCE_BODY;
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        MULTI_REDUCE_INIT_VALUES;

        for (Index_type i = ibegin; i < iend; ++i ) {
          multi_reduce_base_lam(i);
        }

        MULTI_REDUCE_FINALIZE_VALUES;

      }
      stopTimer();

      MULTI_REDUCE_TEARDOWN_VALUES;

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        MULTI_REDUCE_INIT_VALUES_RAJA(RAJA::seq_multi_reduce);

        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            MULTI_REDUCE_BODY;
        });

        MULTI_REDUCE_FINALIZE_VALUES_RAJA(RAJA::seq_multi_reduce);

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  MULTI_REDUCE : Unknown variant id = " << vid << std::endl;
    }

  }

  MULTI_REDUCE_DATA_TEARDOWN;

}

} // end namespace basic
} // end namespace rajaperf
