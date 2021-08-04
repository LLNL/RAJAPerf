//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"

#include <ranges>
#include <algorithm>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace basic
{

//
// Function used in TRAP_INT loop.
//
RAJA_INLINE
Real_type trap_int_func(Real_type x,
                        Real_type y,
                        Real_type xp,
                        Real_type yp)
{
   Real_type denom = (x - xp)*(x - xp) + (y - yp)*(y - yp);
   denom = 1.0/sqrt(denom);
   return denom;
}

void TRAP_INT::runStdParVariant(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRAP_INT_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      auto range = std::views::iota(ibegin, iend);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type sumx = m_sumx_init;

        sumx += std::transform_reduce( std::execution::par_unseq,
                                      std::begin(range), std::end(range),
                                      0.0, std::plus<>(),
                        [=](Index_type i) {
          Real_type x = x0 + i*h;
          return trap_int_func(x, y, xp, yp);
        });
        m_sumx += sumx * h;

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto trapint_base_lam = [=](Index_type i) -> Real_type {
                                Real_type x = x0 + i*h;
                                return trap_int_func(x, y, xp, yp);
                              };

      auto range = std::views::iota(ibegin, iend);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type sumx = m_sumx_init;

        sumx += std::transform_reduce( std::execution::par_unseq,
                                      std::begin(range), std::end(range),
                                      0.0, std::plus<>(), trapint_base_lam);

        m_sumx += sumx * h;

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::seq_reduce, Real_type> sumx(m_sumx_init);

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          TRAP_INT_BODY;
        });

        m_sumx += static_cast<Real_type>(sumx.get()) * h;

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  TRAP_INT : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
