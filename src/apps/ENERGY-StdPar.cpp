//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{


void ENERGY::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto begin = counting_iterator<Index_type>(ibegin);
  auto end   = counting_iterator<Index_type>(iend);

  ENERGY_DATA_SETUP;
  
  auto energy_lam1 = [=](Index_type i) {
                       ENERGY_BODY1;
                     };
  auto energy_lam2 = [=](Index_type i) {
                       ENERGY_BODY2;
                     };
  auto energy_lam3 = [=](Index_type i) {
                       ENERGY_BODY3;
                     };
  auto energy_lam4 = [=](Index_type i) {
                       ENERGY_BODY4;
                     };
  auto energy_lam5 = [=](Index_type i) {
                       ENERGY_BODY5;
                     };
  auto energy_lam6 = [=](Index_type i) {
                       ENERGY_BODY6;
                     };

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          ENERGY_BODY1;
        });

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          ENERGY_BODY2;
        });

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          ENERGY_BODY3;
        });

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          ENERGY_BODY4;
        });
  
        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          ENERGY_BODY5;
        });

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          ENERGY_BODY6;
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
          energy_lam1(i);
        });

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          energy_lam2(i);
        });

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          energy_lam3(i);
        });

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          energy_lam4(i);
        });

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          energy_lam5(i);
        });

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type i) {
          energy_lam6(i);
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  ENERGY : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace apps
} // end namespace rajaperf
