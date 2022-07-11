//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
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

void INDEXLIST::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INDEXLIST_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
#else
      const Index_type n = iend - ibegin;
      ::std::vector<Index_type> tmp_scan(n);
      const int p0 = static_cast<int>(std::min(n, static_cast<Index_type>(omp_get_max_threads())));
      ::std::vector<Index_type> thread_sums(p0);
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Index_type count = 0;

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
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
#else
        #pragma omp parallel num_threads(p0)
        {
          const int p = omp_get_num_threads();
          const int pid = omp_get_thread_num();
          const Index_type step = n / p;
          const Index_type local_begin = pid * step + ibegin;
          const Index_type local_end = (pid == p-1) ? iend : (pid+1) * step + ibegin;

          Index_type local_sum_var = 0;
          for (Index_type i = local_begin; i < local_end; ++i ) {

            Index_type inc = 0;
            if (INDEXLIST_CONDITIONAL) {
              inc = 1;
            }
            tmp_scan[i] = inc;
            local_sum_var += inc;
          }
          thread_sums[pid] = local_sum_var;

          #pragma omp barrier

          Index_type local_count_var = 0;
          for (int ip = 0; ip < pid; ++ip) {
            local_count_var += thread_sums[ip];
          }

          for (Index_type i = local_begin; i < local_end; ++i ) {
            Index_type inc = tmp_scan[i];
            if (inc) {
              list[local_count_var] = i ;
            }
            local_count_var += inc;
          }

          if (pid == p-1) {
            count = local_count_var;
          }
        }
#endif

        m_len = count;

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
      auto indexlist_lam = [=](Index_type i, Index_type count) {
                                  Index_type inc = 0;
                                  if (INDEXLIST_CONDITIONAL) {
                                    list[count] = i ;
                                    inc = 1;
                                  }
                                  return inc;
                                };
#else
      auto indexlist_lam_input = [=](Index_type i) {
                                  Index_type inc = 0;
                                  if (INDEXLIST_CONDITIONAL) {
                                    inc = 1;
                                  }
                                  return inc;
                                };
      auto indexlist_lam_output = [=](Index_type i, Index_type count) {
                                  list[count] = i ;
                                };
      const Index_type n = iend - ibegin;
      ::std::vector<Index_type> tmp_scan(n);
      const int p0 = static_cast<int>(std::min(n, static_cast<Index_type>(omp_get_max_threads())));
      ::std::vector<Index_type> thread_sums(p0);
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Index_type count = 0;

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
        #pragma omp parallel for reduction(inscan, +:count)
        for (Index_type i = ibegin; i < iend; ++i ) {
          #pragma omp scan exclusive(count)
          count += indexlist_lam(i, count);
        }
#else
        #pragma omp parallel num_threads(p0)
        {
          const int p = omp_get_num_threads();
          const int pid = omp_get_thread_num();
          const Index_type step = n / p;
          const Index_type local_begin = pid * step + ibegin;
          const Index_type local_end = (pid == p-1) ? iend : (pid+1) * step + ibegin;

          Index_type local_sum_var = 0;
          for (Index_type i = local_begin; i < local_end; ++i ) {

            Index_type inc = indexlist_lam_input(i);
            tmp_scan[i] = inc;
            local_sum_var += inc;
          }
          thread_sums[pid] = local_sum_var;

          #pragma omp barrier

          Index_type local_count_var = 0;
          for (int ip = 0; ip < pid; ++ip) {
            local_count_var += thread_sums[ip];
          }

          for (Index_type i = local_begin; i < local_end; ++i ) {
            Index_type inc = tmp_scan[i];
            if (inc) {
              indexlist_lam_output(i, local_count_var);
            }
            local_count_var += inc;
          }

          if (pid == p-1) {
            count = local_count_var;
          }
        }
#endif

        m_len = count;

      }
      stopTimer();

      break;
    }

    default : {
      ignore_unused(run_reps, ibegin, iend, x, list);
      getCout() << "\n  INDEXLIST : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace basic
} // end namespace rajaperf
