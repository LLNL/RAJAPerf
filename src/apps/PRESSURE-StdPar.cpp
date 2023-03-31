//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(BUILD_STDPAR)

#include "common/StdParUtils.hpp"

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

  PRESSURE_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                        counting_iterator<Index_type>(ibegin), iend-ibegin,
                        [=](Index_type i) {
          PRESSURE_BODY1;
        });

        std::for_each_n( std::execution::par_unseq,
                        counting_iterator<Index_type>(ibegin), iend-ibegin,
                        [=](Index_type i) {
          PRESSURE_BODY2;
        });

      }
      stopTimer();

      break;
    } 

    case Lambda_StdPar : {

      auto pressure_lam1 = [=](Index_type i) {
                             PRESSURE_BODY1;
                           };
      auto pressure_lam2 = [=](Index_type i) {
                             PRESSURE_BODY2;
                           };
      
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       std::for_each_n( std::execution::par_unseq,
                        counting_iterator<Index_type>(ibegin), iend-ibegin,
                        [=](Index_type i) {
         pressure_lam1(i);
       });

       std::for_each_n( std::execution::par_unseq,
                        counting_iterator<Index_type>(ibegin), iend-ibegin,
                        [=](Index_type i) {
         pressure_lam2(i);
       });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  PRESSURE : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // iend-ibegin namespace apps
} // iend-ibegin namespace rajaperf

#endif  // BUILD_STDPAR

