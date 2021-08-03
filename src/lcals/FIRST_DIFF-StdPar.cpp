//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_DIFF.hpp"

#include "RAJA/RAJA.hpp"

#include <ranges>
#include <algorithm>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void FIRST_DIFF::runStdParVariant(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  FIRST_DIFF_DATA_SETUP;

  auto firstdiff_lam = [=](Index_type i) {
                         x[i] = y[i+1] - y[i];
                       };

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //for (Index_type i = ibegin; i < iend; ++i ) {
        //  x[i] = y[i+1] - y[i];
        //}
        //x[ibegin] = y[ibegin+1] - y[ibegin];
        //std::adjacent_difference( std::execution::par_unseq,
        //                          &y[ibegin], &y[iend], &x[ibegin+1]);

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          firstdiff_lam(i);
        }

      }
      stopTimer();

      break;
    }

#ifdef RAJA_ENABLE_STDPAR
    case RAJA_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), firstdiff_lam);

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  FIRST_DIFF : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace lcals
} // end namespace rajaperf
