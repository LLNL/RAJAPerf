//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST_3LOOP.hpp"

#include "RAJA/RAJA.hpp"

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

#if 0
        Index_type count = 0;
        
        for (Index_type i = ibegin; i < iend+1; ++i ) {
          Index_type inc = counts[i];
          counts[i] = count;
          count += inc;
        }
#else
        // The validation does not notice if the exscan
        // is removed, or otherwise forced to be wrong...
#warning This may be incorrect...
        std::exclusive_scan( std::execution::par_unseq,
                             counts+ibegin, counts+iend+1,
                             counts+ibegin, 0);
#endif

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          INDEXLIST_3LOOP_MAKE_LIST;
        });

        m_len = counts[iend];

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

#if 0
        Index_type count = 0;
        
        for (Index_type i = ibegin; i < iend+1; ++i ) {
          Index_type inc = counts[i];
          counts[i] = count;
          count += inc;
        }
#else
        // The validation does not notice if the exscan
        // is removed, or otherwise forced to be wrong...
#warning This may be incorrect...
        std::exclusive_scan( std::execution::par_unseq,
                             counts+ibegin, counts+iend+1,
                             counts+ibegin, 0);
#endif

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
