//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD_PARTED.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{


void TRIAD_PARTED::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();

  TRIAD_PARTED_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto triad_parted_lam = [=](Index_type i) {
                     TRIAD_PARTED_BODY;
                   };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (size_t p = 1; p < parts.size(); ++p ) {
          const Index_type ibegin = parts[p-1];
          const Index_type iend = parts[p];

          for (Index_type i = ibegin; i < iend; ++i ) {
            TRIAD_PARTED_BODY;
          }
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (size_t p = 1; p < parts.size(); ++p ) {
          const Index_type ibegin = parts[p-1];
          const Index_type iend = parts[p];

          for (Index_type i = ibegin; i < iend; ++i ) {
            triad_parted_lam(i);
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (size_t p = 1; p < parts.size(); ++p ) {
          const Index_type ibegin = parts[p-1];
          const Index_type iend = parts[p];

          RAJA::forall<RAJA::seq_exec>(
            RAJA::RangeSegment(ibegin, iend), triad_parted_lam);
        }

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  TRIAD_PARTED : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace stream
} // end namespace rajaperf
