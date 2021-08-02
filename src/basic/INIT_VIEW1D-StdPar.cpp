//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D.hpp"

#include "RAJA/RAJA.hpp"

#include <ranges>
#include <algorithm>
#include <execution>

#include <iostream>

namespace rajaperf
{
namespace basic
{


void INIT_VIEW1D::runStdParVariant(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INIT_VIEW1D_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      auto range = std::views::iota(ibegin, iend);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                        std::begin(range), std::end(range),
                        [=](Index_type i) {
          INIT_VIEW1D_BODY;
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto range = std::views::iota(ibegin, iend);

      auto initview1d_base_lam = [=](Index_type i) {
                                   INIT_VIEW1D_BODY;
                                 };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                        std::begin(range), std::end(range),
                        [=](Index_type i) {
          initview1d_base_lam(i);
        });

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      INIT_VIEW1D_VIEW_RAJA;

      auto initview1d_lam = [=](Index_type i) {
                              INIT_VIEW1D_BODY_RAJA;
                            };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), initview1d_lam);

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  INIT_VIEW1D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
