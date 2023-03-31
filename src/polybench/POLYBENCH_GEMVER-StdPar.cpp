//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMVER.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_GEMVER::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  POLYBENCH_GEMVER_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), n,
                         [=](Index_type i) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), n,
                           [=](Index_type j) {
            POLYBENCH_GEMVER_BODY1;
          });
        });

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), n,
                         [=](Index_type i) {
          POLYBENCH_GEMVER_BODY2;
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), n,
                           [=,&dot](Index_type j) {
            POLYBENCH_GEMVER_BODY3;
          });
          POLYBENCH_GEMVER_BODY4;
        });

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), n,
                         [=](Index_type i) {
          POLYBENCH_GEMVER_BODY5;
        });

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), n,
                         [=](Index_type i) {
          POLYBENCH_GEMVER_BODY6;
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), n,
                           [=,&dot](Index_type j) {
            POLYBENCH_GEMVER_BODY7;
          });
          POLYBENCH_GEMVER_BODY8;
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto poly_gemver_base_lam1 = [=](Index_type i, Index_type j) {
                                     POLYBENCH_GEMVER_BODY1;
                                   };
      auto poly_gemver_base_lam3 = [=](Index_type i, Index_type j, Real_type &dot) {
                                     POLYBENCH_GEMVER_BODY3;
                                   };
      auto poly_gemver_base_lam4 = [=](Index_type i, Real_type &dot) {
                                     POLYBENCH_GEMVER_BODY4;
                                   };
      auto poly_gemver_base_lam5 = [=](Index_type i) {
                                     POLYBENCH_GEMVER_BODY5;
                                   };
      auto poly_gemver_base_lam7 = [=](Index_type i, Index_type j, Real_type &dot) {
                                     POLYBENCH_GEMVER_BODY7;
                                    };
      auto poly_gemver_base_lam8 = [=](Index_type i, Real_type &dot) {
                                     POLYBENCH_GEMVER_BODY8;
                                   };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), n,
                         [=](Index_type i) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), n,
                           [=](Index_type j) {
            poly_gemver_base_lam1(i, j);
          });
        });

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), n,
                         [=](Index_type i) {
          POLYBENCH_GEMVER_BODY2;
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), n,
                           [=,&dot](Index_type j) {
            poly_gemver_base_lam3(i, j, dot);
          });
          poly_gemver_base_lam4(i, dot);
        });

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), n,
                         [=](Index_type i) {
          poly_gemver_base_lam5(i);
        });

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), n,
                         [=](Index_type i) {
          POLYBENCH_GEMVER_BODY6;
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), n,
                           [=,&dot](Index_type j) {
            poly_gemver_base_lam7(i, j, dot);
          });
          poly_gemver_base_lam8(i, dot);
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  POLYBENCH_GEMVER : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_STDPAR

