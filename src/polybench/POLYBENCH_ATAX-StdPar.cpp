//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ATAX.hpp"

#include "RAJA/RAJA.hpp"

#if defined(BUILD_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

void POLYBENCH_ATAX::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps= getRunReps();

  POLYBENCH_ATAX_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), N,
                         [=](Index_type i) {
          POLYBENCH_ATAX_BODY1;
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), N,
                           [=,&dot](Index_type j) {
            POLYBENCH_ATAX_BODY2;
          });
          POLYBENCH_ATAX_BODY3;
        });

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), N,
                         [=](Index_type j) {
          POLYBENCH_ATAX_BODY4;
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), N,
                           [=,&dot](Index_type i) {
            POLYBENCH_ATAX_BODY5;
          });
          POLYBENCH_ATAX_BODY6;
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto poly_atax_base_lam2 = [=] (Index_type i, Index_type j, Real_type &dot) {
                                   POLYBENCH_ATAX_BODY2;
                                 };
      auto poly_atax_base_lam3 = [=] (Index_type i, Real_type &dot) {
                                   POLYBENCH_ATAX_BODY3;
                                  };
      auto poly_atax_base_lam5 = [=] (Index_type i, Index_type j , Real_type &dot) {
                                   POLYBENCH_ATAX_BODY5;
                                  };
      auto poly_atax_base_lam6 = [=] (Index_type j, Real_type &dot) {
                                   POLYBENCH_ATAX_BODY6;
                                  };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), N,
                         [=](Index_type i) {
          POLYBENCH_ATAX_BODY1;
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), N,
                           [=,&dot](Index_type j) {
            poly_atax_base_lam2(i, j, dot);
          });
          poly_atax_base_lam3(i, dot);
        });

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), N,
                         [=](Index_type j) {
          POLYBENCH_ATAX_BODY4;
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), N,
                           [=,&dot](Index_type i) {
            poly_atax_base_lam5(i, j, dot);
          });
          poly_atax_base_lam6(j, dot);
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  POLYBENCH_ATAX : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf

#endif  // BUILD_STDPAR

