//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void REDUCE_SUM::runStdParVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RUN_STDPAR)
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto begin = counting_iterator<Index_type>(ibegin);
  auto end   = counting_iterator<Index_type>(iend);

  REDUCE_SUM_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type sum = m_sum_init;

#warning needs parallel reduce
        for (Index_type i = ibegin; i < iend; ++i ) {
          REDUCE_SUM_BODY;
        }

        m_sum = sum;

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto reduce_sum_base_lam = [=](Index_type i) {
                                 return x[i];
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type sum = m_sum_init;

#warning needs parallel reduce
        for (Index_type i = ibegin; i < iend; ++i ) {
          sum += reduce_sum_base_lam(i);
        }

        m_sum = sum;

      }
      stopTimer();

      break;
    }

#ifdef RAJA_ENABLE_STDPAR
    case RAJA_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::seq_reduce, Real_type> sum(m_sum_init);

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

#endif
}

} // end namespace algorithm
} // end namespace rajaperf
