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

#define USE_STDPAR_COLLAPSE 1

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_3MM::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  POLYBENCH_3MM_DATA_SETUP;

#ifdef USE_STDPAR_COLLAPSE
  counting_iterator<Index_type> beginIJ(0);
  counting_iterator<Index_type> endIJ(ni*nj);
  counting_iterator<Index_type> beginIL(0);
  counting_iterator<Index_type> endIL(ni*nl);
  counting_iterator<Index_type> beginJL(0);
  counting_iterator<Index_type> endJL(nj*nl);
#else
  counting_iterator<Index_type> beginI(0);
  counting_iterator<Index_type> endI(ni);
  counting_iterator<Index_type> beginL(0);
  counting_iterator<Index_type> endL(nl);
#endif
  counting_iterator<Index_type> beginJ(0);
  counting_iterator<Index_type> endJ(nj);
  counting_iterator<Index_type> beginK(0);
  counting_iterator<Index_type> endK(nk);
  counting_iterator<Index_type> beginM(0);
  counting_iterator<Index_type> endM(nm);

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                       beginIJ, endIJ, [=](Index_type ij) {
            const auto i  = ij / nj;
            const auto j  = ij % nj;
#else
        std::for_each( std::execution::par_unseq,
                       beginI, endI, [=](Index_type i) {
          std::for_each(beginJ, endJ, [=](Index_type j) {
#endif
            POLYBENCH_3MM_BODY1;
            std::for_each(beginK, endK, [=,&dot](Index_type k) {
              POLYBENCH_3MM_BODY2;
            });
            POLYBENCH_3MM_BODY3;
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                       beginJL, endJL, [=](Index_type jl) {
            const auto j  = jl / nl;
            const auto l  = jl % nl;
#else
        std::for_each( std::execution::par_unseq,
                      beginJ, endJ, [=](Index_type j) {
          std::for_each(beginL, endL, [=](Index_type l) {
#endif
            POLYBENCH_3MM_BODY4;
            std::for_each(beginM, endM, [=,&dot](Index_type m) {
              POLYBENCH_3MM_BODY5;
            });
            POLYBENCH_3MM_BODY6;
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                       beginIL, endIL, [=](Index_type il) {
            const auto i  = il / nl;
            const auto l  = il % nl;
#else
        std::for_each( std::execution::par_unseq,
                       beginI, endI, [=](Index_type i) {
          std::for_each(beginL, endL, [=](Index_type l) {
#endif
            POLYBENCH_3MM_BODY7;
            std::for_each(beginJ, endJ, [=,&dot](Index_type j) {
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

      auto poly_3mm_base_lam2 = [=] (Index_type i, Index_type j, Index_type k,
                                     Real_type &dot) {
                                  POLYBENCH_3MM_BODY2;
                                };
      auto poly_3mm_base_lam3 = [=] (Index_type i, Index_type j,
                                     Real_type &dot) {
                                  POLYBENCH_3MM_BODY3;
                                };
      auto poly_3mm_base_lam5 = [=] (Index_type j, Index_type l, Index_type m,
                                     Real_type &dot) {
                                   POLYBENCH_3MM_BODY5;
                                };
      auto poly_3mm_base_lam6 = [=] (Index_type j, Index_type l,
                                     Real_type &dot) {
                                  POLYBENCH_3MM_BODY6;
                                };
      auto poly_3mm_base_lam8 = [=] (Index_type i, Index_type l, Index_type j,
                                     Real_type &dot) {
                                  POLYBENCH_3MM_BODY8;
                                };
      auto poly_3mm_base_lam9 = [=] (Index_type i, Index_type l,
                                     Real_type &dot) {
                                  POLYBENCH_3MM_BODY9;
                                };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                       beginIJ, endIJ, [=](Index_type ij) {
            const auto i  = ij / nj;
            const auto j  = ij % nj;
#else
        std::for_each( std::execution::par_unseq,
                       beginI, endI, [=](Index_type i) {
          std::for_each(beginJ, endJ, [=](Index_type j) {
#endif
            POLYBENCH_3MM_BODY1;
            std::for_each(beginK, endK, [=,&dot](Index_type k) {
              poly_3mm_base_lam2(i, j, k, dot);
            });
            poly_3mm_base_lam3(i, j, dot);
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                       beginJL, endJL, [=](Index_type jl) {
            const auto j  = jl / nl;
            const auto l  = jl % nl;
#else
        std::for_each( std::execution::par_unseq,
                       beginJ, endJ, [=](Index_type j) {
          std::for_each(beginL, endL, [=](Index_type l) {
#endif
            POLYBENCH_3MM_BODY4;
            std::for_each(beginM, endM, [=,&dot](Index_type m) {
              poly_3mm_base_lam5(j, l, m, dot);
            });
            poly_3mm_base_lam6(j, l, dot);
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                       beginIL, endIL, [=](Index_type il) {
            const auto i  = il / nl;
            const auto l  = il % nl;
#else
        std::for_each( std::execution::par_unseq,
                       beginI, endI, [=](Index_type i) {
          std::for_each(beginL, endL, [=](Index_type l) {
#endif
            POLYBENCH_3MM_BODY7;
            std::for_each(beginJ, endJ, [=,&dot](Index_type j) {
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
