//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"

#include <ranges>
#include <algorithm>
#include <execution>

#include <iostream>

//#define USE_STDPAR_COLLAPSE 1

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_FLOYD_WARSHALL::runStdParVariant(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  POLYBENCH_FLOYD_WARSHALL_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

#ifdef USE_STDPAR_COLLAPSE
      auto range2 = std::views::iota((Index_type)0,N*N);
#else
      auto range = std::views::iota((Index_type)0,N);
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                       std::begin(range2), std::end(range2), [=](Index_type ki) {
            const auto k  = ki / N;
            const auto i  = ki % N;
#else
        std::for_each( std::execution::par_unseq,
                        std::begin(range), std::end(range),
                        [=](Index_type k) {
          std::for_each( std::begin(range), std::end(range),
                          [=](Index_type i) {
#endif
            for (Index_type j = 0; j < N; ++j) { 
              POLYBENCH_FLOYD_WARSHALL_BODY;
            }
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto poly_floydwarshall_base_lam = [=](Index_type k, Index_type i, 
                                             Index_type j) {
                                           POLYBENCH_FLOYD_WARSHALL_BODY;
                                         };

#ifdef USE_STDPAR_COLLAPSE
      auto range2 = std::views::iota((Index_type)0,N*N);
#else
      auto range = std::views::iota((Index_type)0,N);
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                       std::begin(range2), std::end(range2), [=](Index_type ki) {
            const auto k  = ki / N;
            const auto i  = ki % N;
#else
        std::for_each( std::execution::par_unseq,
                        std::begin(range), std::end(range),
                        [=](Index_type k) {
          std::for_each( std::begin(range), std::end(range),
                          [=](Index_type i) {
#endif
            for (Index_type j = 0; j < N; ++j) {
              poly_floydwarshall_base_lam(k, i, j);
            }
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

      POLYBENCH_FLOYD_WARSHALL_VIEWS_RAJA; 

      auto poly_floydwarshall_lam = [=](Index_type k, Index_type i, 
                                        Index_type j) {
                                      POLYBENCH_FLOYD_WARSHALL_BODY_RAJA;
                                    };

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                                 RAJA::RangeSegment{0, N},
                                                 RAJA::RangeSegment{0, N}),
          poly_floydwarshall_lam 
        );

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
