//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void GEN_LIN_RECUR::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  GEN_LIN_RECUR_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), N,
                         [=](Index_type k) {
          GEN_LIN_RECUR_BODY1;
        });

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(1), N,
                         [=](Index_type i) {
          GEN_LIN_RECUR_BODY2;
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto genlinrecur_lam1 = [=](Index_type k) {
                                GEN_LIN_RECUR_BODY1;
                              };
      auto genlinrecur_lam2 = [=](Index_type i) {
                                GEN_LIN_RECUR_BODY2;
                              };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), N,
                         [=](Index_type k) {
          genlinrecur_lam1(k);
        });

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(1), N,
                         [=](Index_type i) {
          genlinrecur_lam2(i);
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  GEN_LIN_RECUR : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace lcals
} // end namespace rajaperf
