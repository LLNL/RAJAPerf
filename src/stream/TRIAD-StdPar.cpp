//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{


void TRIAD::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRIAD_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if 0
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          TRIAD_BODY;
        });
#else
        std::transform( std::execution::par_unseq,
                        &b[ibegin], &b[iend], &c[ibegin], &a[ibegin],
                        [=](Real_type b, Real_type c) { return b + alpha * c; });
#endif

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

#if 0
      auto triad_lam = [=](Index_type i) {
                         TRIAD_BODY;
                       };
#else
      auto triad_lam = [=](Real_type b, Real_type c) {
                        return b + alpha * c;
                       };
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if 0
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          triad_lam(i);
        });
#else
        std::transform( std::execution::par_unseq,
                        &b[ibegin], &b[iend], &c[ibegin], &a[ibegin],
                        triad_lam );
#endif
      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  TRIAD : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_STDPAR

