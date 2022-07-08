//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DOT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"
#include <algorithm>
#include <numeric>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace stream
{


void DOT::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DOT_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type dot = m_dot_init;

        dot += std::transform_reduce( std::execution::par_unseq,
                                      &a[ibegin], &a[iend], &b[ibegin],
                                      (Real_type)0);

        m_dot += dot;
      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto dot_base_lam = [=](Index_type i) -> Real_type {
                            return a[i] * b[i];
                          };

      auto begin = counting_iterator<Index_type>(ibegin);
      auto end   = counting_iterator<Index_type>(iend);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type dot = m_dot_init;

        dot += std::transform_reduce( std::execution::par_unseq,
                                      begin,end,
                                      (Real_type)0,
                                      std::plus<Real_type>(),
                                      dot_base_lam);

        m_dot += dot;

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::stdpar_reduce, Real_type> dot(m_dot_init);

        RAJA::forall<RAJA::stdpar_par_unseq_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          DOT_BODY;
        });

        m_dot += static_cast<Real_type>(dot.get());

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      getCout() << "\n  DOT : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace stream
} // end namespace rajaperf
