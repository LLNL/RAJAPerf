//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT_SHARED.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{


void MAT_MAT_SHARED::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  /*
  MAT_MAT_SHARED_DATA_SETUP;

  auto mas_lam = [=](Index_type i) {
                   MAT_MAT_SHARED_BODY;
                 };
  */
  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        /*
        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          MAT_MAT_SHARED_BODY;
        }
        */

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        /*
        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          mas_lam(i);
        }
        */

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        /*
        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), mas_lam);
        */

      }
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  MAT_MAT_SHARED : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
