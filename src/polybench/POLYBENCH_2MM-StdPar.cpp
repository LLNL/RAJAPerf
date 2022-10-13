//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_2MM.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_2MM::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  POLYBENCH_2MM_DATA_SETUP;

#if 0
  auto begin = counting_iterator<Index_type>(0);
  auto end   = counting_iterator<Index_type>(nk);
#endif

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
#if 1
            POLYBENCH_2MM_BODY1;
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(0), nk,
                             [=,&dot](Index_type k) {
              POLYBENCH_2MM_BODY2;
            });
            POLYBENCH_2MM_BODY3;
#else
            tmp[j + i*nj] = std::transform_reduce( std::execution::unseq,
                                                   begin, end,
                                                   (Real_type)0, std::plus<Real_type>(),
                                                   [=] (Index_type k) {
                                                     return alpha * A[k + i*nk] * B[j + k*nj];
                                                   });
#endif
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), ni*nl,
                         [=](Index_type il) {
            const auto i  = il / nl;
            const auto l  = il % nl;
#else
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), ni,
                         [=](Index_type i) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), nl,
                           [=](Index_type l) {
#endif
            POLYBENCH_2MM_BODY4;
            std::for_each_n( std::execution::unseq, 
                             counting_iterator<Index_type>(0), nj,
                             [=,&dot](Index_type j) {
              POLYBENCH_2MM_BODY5;
            });
            POLYBENCH_2MM_BODY6;
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto poly_2mm_base_lam2 =
              [=](Index_type i, Index_type j, Index_type k, Real_type &dot) {
                                  POLYBENCH_2MM_BODY2;
                                };
      auto poly_2mm_base_lam3 =
              [=](Index_type i, Index_type j, Real_type &dot) {
                                  POLYBENCH_2MM_BODY3;
                                };
      auto poly_2mm_base_lam5 =
              [=](Index_type i, Index_type l, Index_type j, Real_type &dot) {
                                  POLYBENCH_2MM_BODY5;
                                };
      auto poly_2mm_base_lam6 =
              [=](Index_type i, Index_type l, Real_type &dot) {
                                  POLYBENCH_2MM_BODY6;
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
            POLYBENCH_2MM_BODY1;
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(0), nk,
                             [=,&dot](Index_type k) {
              poly_2mm_base_lam2(i, j, k, dot);
            });
            poly_2mm_base_lam3(i, j, dot);
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), ni*nl,
                         [=](Index_type il) {
            const auto i  = il / nl;
            const auto l  = il % nl;
#else
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), ni,
                         [=](Index_type i) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), nl,
                           [=](Index_type l) {
#endif
            POLYBENCH_2MM_BODY4;
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(0), nj,
                             [=,&dot](Index_type j) {
              poly_2mm_base_lam5(i, l, j, dot);
            });
            poly_2mm_base_lam6(i, l, dot);
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  POLYBENCH_2MM : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
