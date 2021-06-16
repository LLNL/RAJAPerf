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

#define INDEXLIST_DATA_SETUP_OMP \
  Index_type* counts = new Index_type[getRunSize()+1];

#define INDEXLIST_DATA_TEARDOWN_OMP \
  delete[] counts; counts = nullptr;


void INDEXLIST::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  INDEXLIST_DATA_SETUP;

  switch ( vid ) {

#if _OPENMP >= 201811
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

    case RAJA_OpenMP : {

      INDEXLIST_DATA_SETUP_OMP;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::omp_reduce, Index_type> len(0);

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
          counts[i] = (INDEXLIST_CONDITIONAL) ? 1 : 0;
        });

        RAJA::exclusive_scan_inplace<RAJA::omp_parallel_for_exec>(
            RAJA::make_span(counts+ibegin, iend+1-ibegin));

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
          if (counts[i] != counts[i+1]) {
            list[counts[i]] = i;
            len += 1;
          }
        });

        m_len = len.get();

      }
      stopTimer();

      INDEXLIST_DATA_TEARDOWN_OMP;

      break;
    }

    default : {
      std::cout << "\n  INDEXLIST : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
