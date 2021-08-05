//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMM.hpp"

#include "RAJA/RAJA.hpp"

#include <ranges>
#include <algorithm>
#include <execution>

#include <iostream>

#define USE_STDPAR_COLLAPSE 1

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_GEMM::runStdParVariant(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  POLYBENCH_GEMM_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

#ifdef USE_STDPAR_COLLAPSE
      auto rangeIJ = std::views::iota((Index_type)0, ni*nj);
#else
      auto rangeI = std::views::iota((Index_type)0, ni);
      auto rangeJ = std::views::iota((Index_type)0, nj);
#endif
      auto rangeK = std::views::iota((Index_type)0, nk);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                       std::begin(rangeIJ), std::end(rangeIJ), [=](Index_type ij) {
            const auto i  = ij / nj;
            const auto j  = ij % nj;
#else
        std::for_each( std::execution::par_unseq,
                       std::begin(rangeI), std::end(rangeI), [=](Index_type i) {
          std::for_each( std::begin(rangeJ), std::end(rangeJ), [=](Index_type j) {
#endif
            POLYBENCH_GEMM_BODY1;
            POLYBENCH_GEMM_BODY2;
            std::for_each( std::begin(rangeK), std::end(rangeK), [=,&dot](Index_type k) {
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
      auto poly_gemm_base_lam3 = [=](Index_type i, Index_type j, Index_type k,
                                     Real_type& dot) {
                                   POLYBENCH_GEMM_BODY3;
                                  };
      auto poly_gemm_base_lam4 = [=](Index_type i, Index_type j,
                                     Real_type& dot) {
                                   POLYBENCH_GEMM_BODY4;
                                  };

#ifdef USE_STDPAR_COLLAPSE
      auto rangeIJ = std::views::iota((Index_type)0, ni*nj);
#else
      auto rangeI = std::views::iota((Index_type)0, ni);
      auto rangeJ = std::views::iota((Index_type)0, nj);
#endif
      auto rangeK = std::views::iota((Index_type)0, nk);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                       std::begin(rangeIJ), std::end(rangeIJ), [=](Index_type ij) {
            const auto i  = ij / nj;
            const auto j  = ij % nj;
#else
        std::for_each( std::execution::par_unseq,
                       std::begin(rangeI), std::end(rangeI), [=](Index_type i) {
          std::for_each( std::begin(rangeJ), std::end(rangeJ), [=](Index_type j) {
#endif
            POLYBENCH_GEMM_BODY1;
            poly_gemm_base_lam2(i, j);
            std::for_each( std::begin(rangeK), std::end(rangeK), [=,&dot](Index_type k) {
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

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      POLYBENCH_GEMM_VIEWS_RAJA;

      auto poly_gemm_lam1 = [=](Real_type& dot) {
                                POLYBENCH_GEMM_BODY1_RAJA;
                               };
      auto poly_gemm_lam2 = [=](Index_type i, Index_type j) {
                                POLYBENCH_GEMM_BODY2_RAJA;
                               };
      auto poly_gemm_lam3 = [=](Index_type i, Index_type j, Index_type k, 
                                Real_type& dot) {
                                POLYBENCH_GEMM_BODY3_RAJA;
                               };
      auto poly_gemm_lam4 = [=](Index_type i, Index_type j,
                                Real_type& dot) {
                                POLYBENCH_GEMM_BODY4_RAJA;
                               };

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0, RAJA::Params<0>>,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>>,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<2, RAJA::Segs<0,1,2>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<3, RAJA::Segs<0,1>, RAJA::Params<0>>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
     
          RAJA::make_tuple( RAJA::RangeSegment{0, ni},
                            RAJA::RangeSegment{0, nj},
                            RAJA::RangeSegment{0, nk} ),
          RAJA::tuple<Real_type>{0.0},  // variable for dot

          poly_gemm_lam1,
          poly_gemm_lam2,
          poly_gemm_lam3,
          poly_gemm_lam4

        );

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  POLYBENCH_GEMM : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
