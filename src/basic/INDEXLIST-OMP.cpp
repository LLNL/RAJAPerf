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
