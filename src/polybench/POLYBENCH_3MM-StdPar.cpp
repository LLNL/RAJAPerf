//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_3MM.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_3MM::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  POLYBENCH_3MM_DATA_SETUP;

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
          std::for_each_n( counting_iterator<Index_type>(0), nj,
                           [=](Index_type j) {
#endif
            POLYBENCH_3MM_BODY1;
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(0), nk,
                             [=,&dot](Index_type k) {
              POLYBENCH_3MM_BODY2;
            });
            POLYBENCH_3MM_BODY3;
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(0), nj*nl,
                         [=](Index_type jl) {
            const auto j  = jl / nl;
            const auto l  = jl % nl;
#else
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nj,
                         [=](Index_type j) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), nl,
                           [=](Index_type l) {
#endif
            POLYBENCH_3MM_BODY4;
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(0), nm,
                             [=,&dot](Index_type m) {
              POLYBENCH_3MM_BODY5;
            });
            POLYBENCH_3MM_BODY6;
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
            POLYBENCH_3MM_BODY7;
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(0), nj,
                             [=,&dot](Index_type j) {
              POLYBENCH_3MM_BODY8;
            });
            POLYBENCH_3MM_BODY9;
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto poly_3mm_base_lam2 =
              [=] (Index_type i, Index_type j, Index_type k, Real_type &dot) {
                                  POLYBENCH_3MM_BODY2;
                                };
      auto poly_3mm_base_lam3 =
              [=] (Index_type i, Index_type j, Real_type &dot) {
                                  POLYBENCH_3MM_BODY3;
                                };
      auto poly_3mm_base_lam5 =
              [=] (Index_type j, Index_type l, Index_type m, Real_type &dot) {
                                   POLYBENCH_3MM_BODY5;
                                };
      auto poly_3mm_base_lam6 =
              [=] (Index_type j, Index_type l, Real_type &dot) {
                                  POLYBENCH_3MM_BODY6;
                                };
      auto poly_3mm_base_lam8 =
              [=] (Index_type i, Index_type l, Index_type j, Real_type &dot) {
                                  POLYBENCH_3MM_BODY8;
                                };
      auto poly_3mm_base_lam9 =
              [=] (Index_type i, Index_type l, Real_type &dot) {
                                  POLYBENCH_3MM_BODY9;
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
            POLYBENCH_3MM_BODY1;
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(0), nk,
                             [=,&dot](Index_type k) {
              poly_3mm_base_lam2(i, j, k, dot);
            });
            poly_3mm_base_lam3(i, j, dot);
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                           counting_iterator<Index_type>(0), nj*nl,
                           [=](Index_type jl) {
            const auto j  = jl / nl;
            const auto l  = jl % nl;
#else
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nj,
                         [=](Index_type j) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(0), nl,
                           [=](Index_type l) {
#endif
            POLYBENCH_3MM_BODY4;
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(0), nm,
                             [=,&dot](Index_type m) {
              poly_3mm_base_lam5(j, l, m, dot);
            });
            poly_3mm_base_lam6(j, l, dot);
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
            POLYBENCH_3MM_BODY7;
            std::for_each_n( std::execution::unseq,
                             counting_iterator<Index_type>(0), nj,
                             [=,&dot](Index_type j) {
              poly_3mm_base_lam8(i, l, j, dot);
            });
            poly_3mm_base_lam9(i, l, dot);
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  POLYBENCH_3MM : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
