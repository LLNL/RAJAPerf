//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D_OFFSET.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"
#include <algorithm>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace basic
{


void INIT_VIEW1D_OFFSET::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getActualProblemSize()+1;

  auto begin = counting_iterator<Index_type>(ibegin);
  auto end   = counting_iterator<Index_type>(iend);

  INIT_VIEW1D_OFFSET_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          INIT_VIEW1D_OFFSET_BODY;
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto initview1doffset_base_lam = [=](Index_type i) {
                                         INIT_VIEW1D_OFFSET_BODY;
                                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          initview1doffset_base_lam(i);
        });

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      INIT_VIEW1D_OFFSET_VIEW_RAJA;

      auto initview1doffset_lam = [=](Index_type i) {
                                    INIT_VIEW1D_OFFSET_BODY_RAJA;
                                  };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), initview1doffset_lam);

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  INIT_VIEW1D_OFFSET : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
