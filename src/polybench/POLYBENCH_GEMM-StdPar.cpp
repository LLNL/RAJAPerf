//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_GEMM::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  POLYBENCH_GEMM_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), ni*nj,
                         [=](Index_type ij) {
            const auto i  = ij / nj;
            const auto j  = ij % nj;
#else
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), ni,
                         [=](Index_type i) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), nj,
                           [=](Index_type j) {
#endif
            POLYBENCH_GEMM_BODY1;
            POLYBENCH_GEMM_BODY2;
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(0), nk,
                             [=,&dot](Index_type k) {
               POLYBENCH_GEMM_BODY3;
            });
            POLYBENCH_GEMM_BODY4;
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto poly_gemm_base_lam2 = [=](Index_type i, Index_type j) {
                                   POLYBENCH_GEMM_BODY2;
                                 };
      auto poly_gemm_base_lam3 = [=](Index_type i, Index_type j, Index_type k, Real_type& dot) {
                                   POLYBENCH_GEMM_BODY3;
                                  };
      auto poly_gemm_base_lam4 = [=](Index_type i, Index_type j, Real_type& dot) {
                                   POLYBENCH_GEMM_BODY4;
                                  };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), ni*nj,
                         [=](Index_type ij) {
            const auto i  = ij / nj;
            const auto j  = ij % nj;
#else
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), ni,
                         [=](Index_type i) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), nj,
                           [=](Index_type j) {
#endif
            POLYBENCH_GEMM_BODY1;
            poly_gemm_base_lam2(i, j);
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(0), nk,
                             [=,&dot](Index_type k) {
              poly_gemm_base_lam3(i, j, k, dot);
            });
            poly_gemm_base_lam4(i, j, dot);
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  POLYBENCH_GEMM : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_STDPAR

