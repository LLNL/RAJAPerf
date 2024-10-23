//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MEMCPY.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void MEMCPY::runSeqVariantLibrary(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMCPY_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::memcpy(MEMCPY_STD_ARGS);

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case RAJA_Seq : {

      auto res{getHostResource()}; 

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        res.memcpy(MEMCPY_STD_ARGS);

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  MEMCPY : Unknown variant id = " << vid << std::endl;
    }

  }

}

void MEMCPY::runSeqVariantDefault(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMCPY_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          MEMCPY_BODY;
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto memcpy_lambda = [=](Index_type i) {
                             MEMCPY_BODY;
                           };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          memcpy_lambda(i);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()}; 

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
            MEMCPY_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  MEMCPY : Unknown variant id = " << vid << std::endl;
    }

  }

}

void MEMCPY::runSeqVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if (vid == Base_Seq || vid == RAJA_Seq) {

    if (tune_idx == t) {

      runSeqVariantLibrary(vid);

    }

    t += 1;

  }

  if (tune_idx == t) {

    runSeqVariantDefault(vid);

  }

  t += 1;
}

void MEMCPY::setSeqTuningDefinitions(VariantID vid)
{
  if (vid == Base_Seq || vid == RAJA_Seq) {
    addVariantTuningName(vid, "library");
  }

  addVariantTuningName(vid, "default");
}

} // end namespace algorithm
} // end namespace rajaperf
