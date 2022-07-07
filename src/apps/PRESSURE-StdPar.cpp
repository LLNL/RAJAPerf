//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"
#include <algorithm>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace apps
{


void PRESSURE::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto begin = counting_iterator<Index_type>(ibegin);
  auto end   = counting_iterator<Index_type>(iend);

  PRESSURE_DATA_SETUP;

  auto pressure_lam1 = [=](Index_type i) {
                         PRESSURE_BODY1;
                       };
  auto pressure_lam2 = [=](Index_type i) {
                         PRESSURE_BODY2;
                       };
  
  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          PRESSURE_BODY1;
        });

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          PRESSURE_BODY2;
        });

      }
      stopTimer();

      break;
    } 

    case Lambda_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
         pressure_lam1(i);
       });

       std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
         pressure_lam2(i);
       });

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::seq_region>( [=]() {

          RAJA::forall<RAJA::loop_exec>(
            RAJA::RangeSegment(ibegin, iend), pressure_lam1);

          RAJA::forall<RAJA::loop_exec>(
            RAJA::RangeSegment(ibegin, iend), pressure_lam2);

        }); // end sequential region (for single-source code)

      }
      stopTimer(); 

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  PRESSURE : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace apps
} // end namespace rajaperf
