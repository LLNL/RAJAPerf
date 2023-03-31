//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(BUILD_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

//#define USE_STDPAR_COLLAPSE 1

namespace rajaperf 
{
namespace polybench
{

void POLYBENCH_FLOYD_WARSHALL::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  POLYBENCH_FLOYD_WARSHALL_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < N; ++k) {
#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), N*N,
                         [=](Index_type ji) {
            const auto j  = ji / N;
            const auto i  = ji % N;
#else
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), N,
                         [=](Index_type i) {
          std::for_each_n( std::execution::seq,
                           counting_iterator<Index_type>(0), N,
                           [=](Index_type j) {
#endif
              POLYBENCH_FLOYD_WARSHALL_BODY;
            });
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        }

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto poly_floydwarshall_base_lam = [=](Index_type k, Index_type i, Index_type j) {
                                           POLYBENCH_FLOYD_WARSHALL_BODY;
                                         };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < N; ++k) {
#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), N*N,
                         [=](Index_type ji) {
            const auto j  = ji / N;
            const auto i  = ji % N;
#else
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), N,
                         [=](Index_type i) {
          std::for_each_n( std::execution::seq,
                           counting_iterator<Index_type>(0), N,
                           [=](Index_type j) {
#endif
              poly_floydwarshall_base_lam(k, i, j);
          });
#ifndef USE_STDPAR_COLLAPSE
        });
#endif
       }

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf

#endif  // BUILD_STDPAR

