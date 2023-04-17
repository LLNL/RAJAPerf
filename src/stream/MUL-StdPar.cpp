//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MUL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(BUILD_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{


void MUL::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MUL_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if 0
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          MUL_BODY;
        });
#else
        std::transform( std::execution::par_unseq,
                        &c[ibegin], &c[iend], &b[ibegin],
                        [=](Real_type c) { return alpha * c; });
#endif

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

#if 0
      auto mul_lam = [=](Index_type i) {
                       MUL_BODY;
                     };
#else
      auto mul_lam = [=](Real_type c) {
                       return alpha * c;
                      };
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if 0
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          mul_lam(i);
        });
#else
        std::transform( std::execution::par_unseq,
                        &c[ibegin], &c[iend], &b[ibegin],
                        mul_lam );
#endif
      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  MUL : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace stream
} // end namespace rajaperf

#endif  // BUILD_STDPAR

