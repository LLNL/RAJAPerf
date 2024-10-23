//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MEMSET.hpp"

#include "RAJA/RAJA.hpp"

#include <cstring>
#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void MEMSET::runSeqVariantLibrary(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMSET_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::memset(MEMSET_STD_ARGS);

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        res.memset(MEMSET_STD_ARGS);

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  MEMSET : Unknown variant id = " << vid << std::endl;
    }

  }

}

void MEMSET::runSeqVariantDefault(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMSET_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          MEMSET_BODY;
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto memset_lambda = [=](Index_type i) {
                             MEMSET_BODY;
                           };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          memset_lambda(i);
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
            MEMSET_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  MEMSET : Unknown variant id = " << vid << std::endl;
    }

  }

}

void MEMSET::runSeqVariant(VariantID vid, size_t tune_idx)
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

void MEMSET::setSeqTuningDefinitions(VariantID vid)
{
  if (vid == Base_Seq || vid == RAJA_Seq) {
    addVariantTuningName(vid, "library");
  }

  addVariantTuningName(vid, "default");
}

} // end namespace algorithm
} // end namespace rajaperf
