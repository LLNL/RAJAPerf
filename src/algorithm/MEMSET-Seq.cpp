//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MEMSET_1");
        std::memset(MEMSET_STD_ARGS);
        RP_CALI_SUBKERNEL_END("MEMSET_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MEMSET_1");
        res.memset(MEMSET_STD_ARGS);
        RP_CALI_SUBKERNEL_END("MEMSET_1");

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
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MEMSET_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          MEMSET_BODY;
        }
        RP_CALI_SUBKERNEL_END("MEMSET_1");

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
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MEMSET_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          memset_lambda(i);
        }
        RP_CALI_SUBKERNEL_END("MEMSET_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {
 
      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("MEMSET_1");
        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
            MEMSET_BODY;
        });
        RP_CALI_SUBKERNEL_END("MEMSET_1");

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


void MEMSET::defineSeqVariantTunings()
{

  for (VariantID vid : {Base_Seq, Lambda_Seq, RAJA_Seq}) {

    if (vid == Base_Seq || vid == RAJA_Seq) {

      addVariantTuning<&MEMSET::runSeqVariantLibrary>(
          vid, "library");

    }

    addVariantTuning<&MEMSET::runSeqVariantDefault>(
        vid, "default");

  }

}

} // end namespace algorithm
} // end namespace rajaperf
