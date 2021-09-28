//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#include <ranges>
#include <algorithm>
#include <numeric>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace basic
{


void PI_REDUCE::runStdParVariant(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_REDUCE_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      auto range = std::views::iota(ibegin, iend);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type pi = m_pi_init;
        pi += std::transform_reduce( std::execution::par_unseq,
                                      std::begin(range), std::end(range),
                                      0.0, std::plus<>(),
                        [=](Index_type i) {
          double x = (double(i) + 0.5) * dx;
          return dx / (1.0 + x * x);
        });
        m_pi = 4.0 * pi;

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto pireduce_base_lam = [=](Index_type i) -> Real_type {
                                 double x = (double(i) + 0.5) * dx;
                                 return dx / (1.0 + x * x);
                               };

      auto range = std::views::iota(ibegin, iend);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type pi = m_pi_init;

        pi += std::transform_reduce( std::execution::par_unseq,
                                      std::begin(range), std::end(range),
                                      0.0, std::plus<>(), pireduce_base_lam);

        m_pi = 4.0 * pi;

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
  
        RAJA::ReduceSum<RAJA::seq_reduce, Real_type> pi(m_pi_init); 

        RAJA::forall<RAJA::loop_exec>( RAJA::RangeSegment(ibegin, iend), 
          [=](Index_type i) {
            PI_REDUCE_BODY;
        });

        m_pi = 4.0 * pi.get();

      }
      stopTimer();

      break;
    }
#endif

    default : {
      std::cout << "\n  PI_REDUCE : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
