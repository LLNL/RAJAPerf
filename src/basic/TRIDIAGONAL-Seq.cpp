//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAGONAL.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void TRIDIAGONAL::runSeqVariant(VariantID vid, size_t /*tune_idx*/)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRIDIAGONAL_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      TRIDIAGONAL_TEMP_DATA_SETUP_LOCAL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          TRIDIAGONAL_LOCAL_DATA_SETUP;
          TRIDIAGONAL_BODY_FORWARD_TEMP_LOCAL;
          TRIDIAGONAL_BODY_BACKWARD_TEMP_LOCAL;
        }

      }
      stopTimer();

      TRIDIAGONAL_TEMP_DATA_TEARDOWN_LOCAL;

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      TRIDIAGONAL_TEMP_DATA_SETUP_LOCAL;

      auto triad_lam = [=](Index_type i) {
                         TRIDIAGONAL_LOCAL_DATA_SETUP;
                         TRIDIAGONAL_BODY_FORWARD_TEMP_LOCAL;
                         TRIDIAGONAL_BODY_BACKWARD_TEMP_LOCAL;
                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          triad_lam(i);
        }

      }
      stopTimer();

      TRIDIAGONAL_TEMP_DATA_TEARDOWN_LOCAL;

      break;
    }

    case RAJA_Seq : {

      TRIDIAGONAL_TEMP_DATA_SETUP_LOCAL;

      auto triad_lam = [=](Index_type i) {
                         TRIDIAGONAL_LOCAL_DATA_SETUP;
                         TRIDIAGONAL_BODY_FORWARD_TEMP_LOCAL;
                         TRIDIAGONAL_BODY_BACKWARD_TEMP_LOCAL;
                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), triad_lam);

      }
      stopTimer();

      TRIDIAGONAL_TEMP_DATA_TEARDOWN_LOCAL;

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  TRIDIAGONAL : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace stream
} // end namespace rajaperf
