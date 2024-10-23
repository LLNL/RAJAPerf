//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SCAN.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>
#include <vector>

namespace rajaperf
{
namespace algorithm
{

void SCAN::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SCAN_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
#else
      const Index_type n = iend - ibegin;
      const int p0 = static_cast<int>(std::min(n, static_cast<Index_type>(omp_get_max_threads())));
      ::std::vector<Real_type> thread_sums(p0);
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        SCAN_PROLOGUE;

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
        #pragma omp parallel for reduction(inscan, +:scan_var)
        for (Index_type i = ibegin; i < iend; ++i ) {
          y[i] = scan_var;
          #pragma omp scan exclusive(scan_var)
          scan_var += x[i];
        }
#else
        #pragma omp parallel num_threads(p0)
        {
          const int p = omp_get_num_threads();
          const int pid = omp_get_thread_num();
          const Index_type step = n / p;
          const Index_type local_begin = pid * step + ibegin;
          const Index_type local_end = (pid == p-1) ? iend : (pid+1) * step + ibegin;

          Real_type local_scan_var = (pid == 0) ? scan_var : 0;
          for (Index_type i = local_begin; i < local_end; ++i ) {
            y[i] = local_scan_var;
            local_scan_var += x[i];
          }
          thread_sums[pid] = local_scan_var;

          #pragma omp barrier

          if (pid != 0) {

            Real_type prev_sum = 0;
            for (int ip = 0; ip < pid; ++ip) {
              prev_sum += thread_sums[ip];
            }

            for (Index_type i = local_begin; i < local_end; ++i ) {
              y[i] += prev_sum;
            }
          }
        }
#endif

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
        auto scan_lam = [=](Index_type i, Real_type scan_var) {
                          y[i] = scan_var;
                          return x[i];
                        };
#else
        auto scan_lam_input = [=](Index_type i) {
                          return x[i];
                        };
        auto scan_lam_sum_output = [=](Index_type i, Real_type sum_var) {
                          y[i] += sum_var;
                        };
        auto scan_lam_output = [=](Index_type i, Real_type scan_var) {
                          y[i] = scan_var;
                        };

        const Index_type n = iend - ibegin;
        const int p0 = static_cast<int>(std::min(n, static_cast<Index_type>(omp_get_max_threads())));
        ::std::vector<Real_type> thread_sums(p0);
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {


        SCAN_PROLOGUE;

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
        #pragma omp parallel for reduction(inscan, +:scan_var)
        for (Index_type i = ibegin; i < iend; ++i ) {
          #pragma omp scan exclusive(scan_var)
          scan_var += scan_lam(i, scan_var);
        }
#else
        #pragma omp parallel num_threads(p0)
        {
          const int p = omp_get_num_threads();
          const int pid = omp_get_thread_num();
          const Index_type step = n / p;
          const Index_type local_begin = pid * step + ibegin;
          const Index_type local_end = (pid == p-1) ? iend : (pid+1) * step + ibegin;

          Real_type local_scan_var = (pid == 0) ? scan_var : 0;
          for (Index_type i = local_begin; i < local_end; ++i ) {
            scan_lam_output(i, local_scan_var);
            local_scan_var += scan_lam_input(i);
          }
          thread_sums[pid] = local_scan_var;

          #pragma omp barrier

          if (pid != 0) {
            Real_type prev_sum = 0;
            for (int ip = 0; ip < pid; ++ip) {
              prev_sum += thread_sums[ip];
            }

            for (Index_type i = local_begin; i < local_end; ++i ) {
              scan_lam_sum_output(i, prev_sum);
            }
          }
        }
#endif

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::exclusive_scan<RAJA::omp_parallel_for_exec>(res, RAJA_SCAN_ARGS);

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  SCAN : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace algorithm
} // end namespace rajaperf
