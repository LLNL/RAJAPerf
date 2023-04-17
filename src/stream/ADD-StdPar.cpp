//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ADD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(BUILD_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{


void ADD::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ADD_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if 0
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          ADD_BODY;
        });
#else
        std::transform( std::execution::par_unseq,
                        &a[ibegin], &a[iend], &b[ibegin], &c[ibegin],
                        [=](Real_type a, Real_type b) { return a + b; });
#endif

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

#if 0
      auto add_lam = [=](Index_type i) {
                       ADD_BODY;
                     };
#else
      auto add_lam = [=](Real_type a, Real_type b) {
                       return a + b;
                      };
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if 0
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          add_lam(i);
        });
#else
        std::transform( std::execution::par_unseq,
                        &a[ibegin], &a[iend], &b[ibegin], &c[ibegin],
                        add_lam );
#endif
      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  ADD : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace stream
} // end namespace rajaperf

#endif  // BUILD_STDPAR

