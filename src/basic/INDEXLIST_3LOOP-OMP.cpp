//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
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
  Index_type* counts = new Index_type[iend+1];

#define INDEXLIST_3LOOP_DATA_TEARDOWN_OMP \
  delete[] counts; counts = nullptr;


void INDEXLIST_3LOOP::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INDEXLIST_3LOOP_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      INDEXLIST_3LOOP_DATA_SETUP_OMP;

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
#else
      const Index_type n = iend+1 - ibegin;
      const int p0 = static_cast<int>(std::min(n, static_cast<Index_type>(omp_get_max_threads())));
      ::std::vector<Index_type> thread_counts(p0);
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
        }

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
        Index_type count = 0;
        #pragma omp parallel for reduction(inscan, +:count)
        for (Index_type i = ibegin; i < iend+1; ++i ) {
          Index_type inc = counts[i];
          counts[i] = count;
          #pragma omp scan exclusive(count)
          count += inc;
        }
#else
        #pragma omp parallel num_threads(p0)
        {
          const int p = omp_get_num_threads();
          const int pid = omp_get_thread_num();
          const Index_type step = n / p;
          const Index_type local_begin = pid * step + ibegin;
          const Index_type local_end = (pid == p-1) ? iend+1 : (pid+1) * step + ibegin;

          Index_type local_count = 0;
          for (Index_type i = local_begin; i < local_end; ++i ) {
            Index_type inc = counts[i];
            counts[i] = local_count;
            local_count += inc;
          }
          thread_counts[pid] = local_count;

          #pragma omp barrier

          if (pid != 0) {

            Index_type prev_count = 0;
            for (int ip = 0; ip < pid; ++ip) {
              prev_count += thread_counts[ip];
            }

            for (Index_type i = local_begin; i < local_end; ++i ) {
              counts[i] += prev_count;
            }
          }
        }
#endif

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

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
#else
      const Index_type n = iend+1 - ibegin;
      const int p0 = static_cast<int>(std::min(n, static_cast<Index_type>(omp_get_max_threads())));
      ::std::vector<Index_type> thread_counts(p0);
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          indexlist_conditional_lam(i);
        }

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
        Index_type count = 0;
        #pragma omp parallel for reduction(inscan, +:count)
        for (Index_type i = ibegin; i < iend+1; ++i ) {
          Index_type inc = counts[i];
          counts[i] = count;
          #pragma omp scan exclusive(count)
          count += inc;
        }
#else
        #pragma omp parallel num_threads(p0)
        {
          const int p = omp_get_num_threads();
          const int pid = omp_get_thread_num();
          const Index_type step = n / p;
          const Index_type local_begin = pid * step + ibegin;
          const Index_type local_end = (pid == p-1) ? iend+1 : (pid+1) * step + ibegin;

          Index_type local_count = 0;
          for (Index_type i = local_begin; i < local_end; ++i ) {
            Index_type inc = counts[i];
            counts[i] = local_count;
            local_count += inc;
          }
          thread_counts[pid] = local_count;

          #pragma omp barrier

          if (pid != 0) {

            Index_type prev_count = 0;
            for (int ip = 0; ip < pid; ++ip) {
              prev_count += thread_counts[ip];
            }

            for (Index_type i = local_begin; i < local_end; ++i ) {
              counts[i] += prev_count;
            }
          }
        }
#endif

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

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      INDEXLIST_3LOOP_DATA_SETUP_OMP;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
          counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
        });

        RAJA::exclusive_scan_inplace<RAJA::omp_parallel_for_exec>(
            RAJA::make_span(counts+ibegin, iend+1-ibegin));

        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
          if (counts[i] != counts[i+1]) {
            list[counts[i]] = i;
          }
        });

        m_len = counts[iend];

      }
      stopTimer();

      INDEXLIST_3LOOP_DATA_TEARDOWN_OMP;

      break;
    }

    default : {
      getCout() << "\n  INDEXLIST_3LOOP : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace basic
} // end namespace rajaperf
