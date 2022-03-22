//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAGONAL_PAR.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void TRIDIAGONAL_PAR::runOpenMPVariant(VariantID vid, size_t /*tune_idx*/)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRIDIAGONAL_PAR_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
        {
          TRIDIAGONAL_PAR_TEMP_DATA_SETUP_LOCAL;

          #pragma omp for
          for (Index_type i = ibegin; i < iend; ++i ) {
            TRIDIAGONAL_PAR_LOCAL_DATA_SETUP;
            TRIDIAGONAL_PAR_BODY_FORWARD_TEMP_LOCAL;
            TRIDIAGONAL_PAR_BODY_BACKWARD_TEMP_LOCAL;
          }

          TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_LOCAL;
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
        {
          TRIDIAGONAL_PAR_TEMP_DATA_SETUP_LOCAL;

          auto tridiagonal_lam = [=](Index_type i) {
                             TRIDIAGONAL_PAR_LOCAL_DATA_SETUP;
                             TRIDIAGONAL_PAR_BODY_FORWARD_TEMP_LOCAL;
                             TRIDIAGONAL_PAR_BODY_BACKWARD_TEMP_LOCAL;
                           };

          #pragma omp for
          for (Index_type i = ibegin; i < iend; ++i ) {
            tridiagonal_lam(i);
          }

          TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_LOCAL;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

          TRIDIAGONAL_PAR_TEMP_DATA_SETUP_LOCAL;

          auto tridiagonal_lam = [=](Index_type i) {
                             TRIDIAGONAL_PAR_LOCAL_DATA_SETUP;
                             TRIDIAGONAL_PAR_BODY_FORWARD_TEMP_LOCAL;
                             TRIDIAGONAL_PAR_BODY_BACKWARD_TEMP_LOCAL;
                           };

          RAJA::forall< RAJA::omp_for_exec >(
            RAJA::RangeSegment(ibegin, iend), tridiagonal_lam);

          TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_LOCAL;

        }); // end omp parallel region

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  TRIDIAGONAL_PAR : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace stream
} // end namespace rajaperf
