//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "COPY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(BUILD_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{


void COPY::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  COPY_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if 0
        std::copy( std::execution::par_unseq,
                   &a[ibegin], &a[iend], &c[ibegin]);
#else
        std::transform( std::execution::par_unseq,
                        &a[ibegin], &a[iend], &c[ibegin],
                        [=](Real_type a) { return a; });
#endif

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

#if 1
      auto copy_lam = [=](Real_type a) {
                       return a;
                      };
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if 0
        std::copy( std::execution::par_unseq,
                   &a[ibegin], &a[iend], &c[ibegin]);
#else
        std::transform( std::execution::par_unseq,
                        &a[ibegin], &a[iend], &c[ibegin],
                        copy_lam );
#endif

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  COPY : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace stream
} // end namespace rajaperf

#endif  // BUILD_STDPAR

