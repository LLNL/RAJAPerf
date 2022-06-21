//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void REDUCE_SUM::runSeqVariantKahan(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type sum = sum_init;
        Real_type ckahan = 0.0;

        for (Index_type i = ibegin; i < iend; ++i ) {
          Real_type y = x[i] - ckahan;
          volatile Real_type t = sum + y;
          volatile Real_type z = t - sum;
          ckahan = z - y;
          sum = t;
        }

        m_sum = sum;

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  REDUCE_SUM : Unknown variant id = " << vid << std::endl;
    }

  }

}

void REDUCE_SUM::runSeqVariantDefault(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type sum = sum_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          REDUCE_SUM_BODY;
        }

        m_sum = sum;

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto reduce_sum_base_lam = [=](Index_type i) {
                                 return x[i];
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type sum = sum_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          sum += reduce_sum_base_lam(i);
        }

        m_sum = sum;

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::seq_reduce, Real_type> sum(sum_init);

        RAJA::forall<RAJA::loop_exec>( RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
            REDUCE_SUM_BODY;
        });

        m_sum = sum.get();

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  REDUCE_SUM : Unknown variant id = " << vid << std::endl;
    }

  }

}

void REDUCE_SUM::runSeqVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_Seq ) {

    if (tune_idx == t) {

      runSeqVariantKahan(vid);

    }

    t += 1;

  }

  if (tune_idx == t) {

    runSeqVariantDefault(vid);

  }

  t += 1;

}

void REDUCE_SUM::setSeqTuningDefinitions(VariantID vid)
{
  if ( vid == Base_Seq ) {

    addVariantTuningName(vid, "kahan");

  }

  addVariantTuningName(vid, "default");

}

} // end namespace algorithm
} // end namespace rajaperf
