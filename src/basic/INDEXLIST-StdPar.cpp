//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

void INDEXLIST::runStdParVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RUN_STDPAR)
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INDEXLIST_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      auto counts = std::vector<Index_type>(iend+1,0);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if 0
        Index_type count = 0;

#warning needs parallel something
        for (Index_type i = ibegin; i < iend; ++i ) {
          if ( x[i] < 0.0 ) {
            list[count++] = i;
            y[i] = 1;
          }
        }

        m_len = count;
#else
        std::transform_exclusive_scan( //std::execution:seq,
                                       &x[ibegin], &x[iend],
                                       &counts[0], 0,
                                       std::plus<Index_type>{},
                                       [=](Real_type x){ return (x < 0.0); });

        std::for_each_n( //std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          if (counts[i] != counts[i+1]) { \
            list[counts[i]] = i;
          }
        });

        m_len = counts[iend+1];
#endif

        if (irep == 0) {
            //printf("\n\n%d\n",counts[iend]);
            //for (Index_type i = ibegin, j=0; i < iend && j<count ; ++i, ++j ) {
            for (Index_type i = ibegin; i < iend ; ++i) {
                printf("%6d: x=%9.6f counts=%6d list=%6d\n",i,x[i],counts[i],list[i]);
            }
            printf("\n\n");

        }

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto indexlist_base_lam = [=](Index_type i, Index_type& count) {
                                 INDEXLIST_BODY
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Index_type count = 0;

#warning needs parallel inscan
        for (Index_type i = ibegin; i < iend; ++i ) {
          indexlist_base_lam(i, count);
        }

        m_len = count;

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  INDEXLIST : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
