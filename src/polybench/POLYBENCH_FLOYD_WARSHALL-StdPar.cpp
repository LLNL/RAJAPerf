//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"

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

#ifdef USE_STDPAR_COLLAPSE
  counting_iterator<Index_type> begin2(0);
  counting_iterator<Index_type> end2(N*N);
#else
  counting_iterator<Index_type> begin(0);
  counting_iterator<Index_type> end(N);
#endif

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                       begin2, end2, [=](Index_type ki) {
            const auto k  = ki / N;
            const auto i  = ki % N;
#else
        std::for_each( std::execution::par_unseq,
                       begin, end,
                       [=](Index_type k) {
          std::for_each(begin, end,
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

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                       begin2, end2, [=](Index_type ki) {
            const auto k  = ki / N;
            const auto i  = ki % N;
#else
        std::for_each( std::execution::par_unseq,
                       begin, end,
                       [=](Index_type k) {
          std::for_each(begin, end,
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

    default : {
      getCout() << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace polybench
} // end namespace rajaperf
