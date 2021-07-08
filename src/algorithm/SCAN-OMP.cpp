//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SCAN.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void SCAN::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  SCAN_DATA_SETUP;

  switch ( vid ) {

#if _OPENMP >= 201811
    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        SCAN_PROLOGUE;
        #pragma omp parallel for reduction(inscan, +:scan_var)
        for (Index_type i = ibegin; i < iend; ++i ) {
          y[i] = scan_var;
          #pragma omp scan exclusive(scan_var)
          scan_var += x[i];
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        SCAN_PROLOGUE;
        auto scan_lam = [=](Index_type i, Real_type& scan_var) {
                          y[i] = scan_var;
                          return x[i];
                        };
        #pragma omp parallel for reduction(inscan, +:scan_var)
        for (Index_type i = ibegin; i < iend; ++i ) {
          #pragma omp scan exclusive(scan_var)
          scan_var += scan_lam(i, scan_var);
        }

      }
      stopTimer();

      break;
    }
#endif

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::exclusive_scan<RAJA::omp_parallel_for_exec>(RAJA_SCAN_ARGS);

      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  SCAN : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace algorithm
} // end namespace rajaperf
