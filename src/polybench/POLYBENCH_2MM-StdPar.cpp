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

#define USE_STDPAR_COLLAPSE 1

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_2MM::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  POLYBENCH_2MM_DATA_SETUP;

#ifdef USE_STDPAR_COLLAPSE
  counting_iterator<Index_type> beginIJ(0);
  counting_iterator<Index_type> endIJ(ni*nj);
  counting_iterator<Index_type> beginIL(0);
  counting_iterator<Index_type> endIL(ni*nl);
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
            POLYBENCH_2MM_BODY1;
            std::for_each(beginK, endK, [=,&dot](Index_type k) {
              POLYBENCH_2MM_BODY2;
            });
            POLYBENCH_2MM_BODY3;
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
            POLYBENCH_2MM_BODY4;
            std::for_each(beginJ, endJ, [=,&dot](Index_type j) {
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

      auto poly_2mm_base_lam2 = [=](Index_type i, Index_type j,
                                    Index_type k, Real_type &dot) {
                                  POLYBENCH_2MM_BODY2;
                                };
      auto poly_2mm_base_lam3 = [=](Index_type i, Index_type j,
                                    Real_type &dot) {
                                  POLYBENCH_2MM_BODY3;
                                };
      auto poly_2mm_base_lam5 = [=](Index_type i, Index_type l,
                                    Index_type j, Real_type &dot) {
                                  POLYBENCH_2MM_BODY5;
                                };
      auto poly_2mm_base_lam6 = [=](Index_type i, Index_type l,
                                    Real_type &dot) {
                                  POLYBENCH_2MM_BODY6;
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
            POLYBENCH_2MM_BODY1;
            std::for_each(beginK, endK, [=,&dot](Index_type k) {
              poly_2mm_base_lam2(i, j, k, dot);
            });
            poly_2mm_base_lam3(i, j, dot);
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
            POLYBENCH_2MM_BODY4;
            std::for_each(beginJ, endJ, [=,&dot](Index_type j) {
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

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      POLYBENCH_2MM_VIEWS_RAJA;

      auto poly_2mm_lam1 = [=](Real_type &dot) {
                             POLYBENCH_2MM_BODY1_RAJA;
                           };
      auto poly_2mm_lam2 = [=](Index_type i, Index_type j, Index_type k, 
                               Real_type &dot) {
                             POLYBENCH_2MM_BODY2_RAJA;
                           };
      auto poly_2mm_lam3 = [=](Index_type i, Index_type j,
                               Real_type &dot) {
                             POLYBENCH_2MM_BODY3_RAJA;
                           };
      auto poly_2mm_lam4 = [=](Real_type &dot) {
                             POLYBENCH_2MM_BODY4_RAJA;
                           };
      auto poly_2mm_lam5 = [=](Index_type i, Index_type l, Index_type j, 
                               Real_type &dot) {
                             POLYBENCH_2MM_BODY5_RAJA;
                           };
      auto poly_2mm_lam6 = [=](Index_type i, Index_type l,
                               Real_type &dot) {
                             POLYBENCH_2MM_BODY6_RAJA;
                           };

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0, RAJA::Params<0>>,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<1, RAJA::Segs<0,1,2>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0,1>, RAJA::Params<0>>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>( 
          RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nj},
                           RAJA::RangeSegment{0, nk}),
          RAJA::tuple<Real_type>{0.0},

          poly_2mm_lam1, 
          poly_2mm_lam2, 
          poly_2mm_lam3
        );

        RAJA::kernel_param<EXEC_POL>( 
          RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nl},
                           RAJA::RangeSegment{0, nj}),
          RAJA::tuple<Real_type>{0.0},

          poly_2mm_lam4, 
          poly_2mm_lam5, 
          poly_2mm_lam6

        );

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      getCout() << "\n  POLYBENCH_2MM : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
