//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GESUMMV.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_GESUMMV::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps= getRunReps();

  POLYBENCH_GESUMMV_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), N,
                         [=](Index_type i) {
          POLYBENCH_GESUMMV_BODY1;
          std::for_each_n( counting_iterator<Index_type>(0), N,
                           [=,&tmpdot,&ydot](Index_type j) {
            POLYBENCH_GESUMMV_BODY2;
          });
          POLYBENCH_GESUMMV_BODY3;
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto poly_gesummv_base_lam2 = [=](Index_type i, Index_type j, Real_type& tmpdot, Real_type& ydot) {
                                      POLYBENCH_GESUMMV_BODY2;
                                    };
      auto poly_gesummv_base_lam3 = [=](Index_type i, Real_type& tmpdot, Real_type& ydot) {
                                      POLYBENCH_GESUMMV_BODY3;
                                    };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), N,
                         [=](Index_type i) {
          POLYBENCH_GESUMMV_BODY1;
          std::for_each_n( counting_iterator<Index_type>(0), N,
                           [=,&tmpdot,&ydot](Index_type j) {
            poly_gesummv_base_lam2(i, j, tmpdot, ydot);
          });
          poly_gesummv_base_lam3(i, tmpdot, ydot);
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  POLYBENCH_GESUMMV : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // N namespace polybench
} // N namespace rajaperf
