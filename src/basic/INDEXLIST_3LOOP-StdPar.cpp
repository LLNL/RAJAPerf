//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST_3LOOP.hpp"

#include "RAJA/RAJA.hpp"

#if defined(BUILD_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define INDEXLIST_3LOOP_DATA_SETUP_StdPar \
  Index_type* counts = new Index_type[iend+1];

#define INDEXLIST_3LOOP_DATA_TEARDOWN_StdPar \
  delete[] counts; counts = nullptr;



void INDEXLIST_3LOOP::runStdParVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INDEXLIST_3LOOP_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      INDEXLIST_3LOOP_DATA_SETUP_StdPar;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
        });

        // The validation does not notice if the exscan
        // is removed, or otherwise forced to be wrong.
        // Using brute-force validation (see below):
        // Intel and GCC output 0s when any execution policy is used.
        // NVHPC (GPU) is fine.
        std::exclusive_scan(
#ifdef __NVCOMPILER
                             std::execution::par_unseq,
#endif
                             counts+ibegin, counts+iend+1,
                             counts+ibegin, 0);

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          INDEXLIST_3LOOP_MAKE_LIST;
        });

        m_len = counts[iend];

#if BRUTE_FORCE_VALIDATION
        for (Index_type i = ibegin; i < iend+1; ++i ) {
          std::cout << "C: " << i << "," << counts[i] << "\n";
        }
#endif
      }
      stopTimer();

      INDEXLIST_3LOOP_DATA_TEARDOWN_StdPar;

      break;
    }

    case Lambda_StdPar : {

      INDEXLIST_3LOOP_DATA_SETUP_StdPar;

      auto indexlist_conditional_lam = [=](Index_type i) {
                                  counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
                                };

      auto indexlist_make_list_lam = [=](Index_type i) {
                                  INDEXLIST_3LOOP_MAKE_LIST;
                                };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          indexlist_conditional_lam(i);
        });

        // See comments above...
        std::exclusive_scan(
#ifdef __NVCOMPILER
                             std::execution::par_unseq,
#endif
                             counts+ibegin, counts+iend+1,
                             counts+ibegin, 0);

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          indexlist_make_list_lam(i);
        });

        m_len = counts[iend];

      }
      stopTimer();

      INDEXLIST_3LOOP_DATA_TEARDOWN_StdPar;

      break;
    }

    default : {
      getCout() << "\n  INDEXLIST_3LOOP : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf

#endif  // BUILD_STDPAR

