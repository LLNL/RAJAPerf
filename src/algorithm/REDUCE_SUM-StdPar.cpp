//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(BUILD_STDPAR)

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

        sum += std::reduce( std::execution::par_unseq,
                            x+ibegin, x+iend,
                            Real_type(0), std::plus<Real_type>() );

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

        sum += std::transform_reduce( std::execution::par_unseq,
                                      begin, end,
                                      Real_type(0), std::plus<Real_type>(), reduce_sum_base_lam);

        m_sum = sum;

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  REDUCE_SUM : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // BUILD_STDPAR

