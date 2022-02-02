//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

void INDEXLIST::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INDEXLIST_DATA_SETUP;

  switch ( vid ) {

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP_SCAN)
    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Index_type count = 0;

        #pragma omp parallel for reduction(inscan, +:count)
        for (Index_type i = ibegin; i < iend; ++i ) {
          Index_type inc = 0;
          if (INDEXLIST_CONDITIONAL) {
            list[count] = i ;
            inc = 1;
          }
          #pragma omp scan exclusive(count)
          count += inc;
        }

        m_len = count;

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto indexlist_base_lam = [=](Index_type i, Index_type& count) {
                                  Index_type inc = 0;
                                  if (INDEXLIST_CONDITIONAL) {
                                    list[count] = i ;
                                    inc = 1;
                                  }
                                  return inc;
                                };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Index_type count = 0;

        #pragma omp parallel for reduction(inscan, +:count)
        for (Index_type i = ibegin; i < iend; ++i ) {
          #pragma omp scan exclusive(count)
          count += indexlist_base_lam(i, count);
        }

        m_len = count;

      }
      stopTimer();

      break;
    }
#endif

    default : {
      ignore_unused(run_reps, ibegin, iend, x, list);
      std::cout << "\n  INDEXLIST : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
