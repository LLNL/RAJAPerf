//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST_3LOOP.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define INDEXLIST_3LOOP_DATA_SETUP_OMP \
  Index_type* counts = new Index_type[getActualProblemSize()+1];

#define INDEXLIST_3LOOP_DATA_TEARDOWN_OMP \
  delete[] counts; counts = nullptr;


void INDEXLIST_3LOOP::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INDEXLIST_3LOOP_DATA_SETUP;

  switch ( vid ) {

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP_SCAN)
    case Base_OpenMP : {

      INDEXLIST_3LOOP_DATA_SETUP_OMP;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
        }

        Index_type count = 0;

        #pragma omp parallel for reduction(inscan, +:count)
        for (Index_type i = ibegin; i < iend+1; ++i ) {
          Index_type inc = counts[i];
          counts[i] = count;
          #pragma omp scan exclusive(count)
          count += inc;
        }

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          INDEXLIST_3LOOP_MAKE_LIST;
        }

        m_len = counts[iend];

      }
      stopTimer();

      INDEXLIST_3LOOP_DATA_TEARDOWN_OMP;

      break;
    }

    case Lambda_OpenMP : {

      INDEXLIST_3LOOP_DATA_SETUP_OMP;

      auto indexlist_conditional_lam = [=](Index_type i) {
                                  counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
                                };

      auto indexlist_make_list_lam = [=](Index_type i) {
                                  INDEXLIST_3LOOP_MAKE_LIST;
                                };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          indexlist_conditional_lam(i);
        }

        Index_type count = 0;

        #pragma omp parallel for reduction(inscan, +:count)
        for (Index_type i = ibegin; i < iend+1; ++i ) {
          Index_type inc = counts[i];
          counts[i] = count;
          #pragma omp scan exclusive(count)
          count += inc;
        }

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          indexlist_make_list_lam(i);
        }

        m_len = counts[iend];

      }
      stopTimer();

      INDEXLIST_3LOOP_DATA_TEARDOWN_OMP;

      break;
    }
#endif

    case RAJA_OpenMP : {

      INDEXLIST_3LOOP_DATA_SETUP_OMP;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::omp_reduce, Index_type> len(0);

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
          counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
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

      INDEXLIST_3LOOP_DATA_TEARDOWN_OMP;

      break;
    }

    default : {
      std::cout << "\n  INDEXLIST_3LOOP : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
