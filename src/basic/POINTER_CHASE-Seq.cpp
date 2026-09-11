//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POINTER_CHASE.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

void POINTER_CHASE::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POINTER_CHASE_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto pointer_chase_lam = [=](Index_type) {
    POINTER_CHASE_BODY;
  };
#endif

  switch (vid) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps;
           RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POINTER_CHASE_1");
        POINTER_CHASE_BODY;
        RP_CALI_SUBKERNEL_END("POINTER_CHASE_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps;
           RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POINTER_CHASE_1");
        pointer_chase_lam(0);
        RP_CALI_SUBKERNEL_END("POINTER_CHASE_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps;
           RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POINTER_CHASE_1");
        RAJA::forall<RAJA::seq_exec>(res, RAJA::RangeSegment(0, 1),
                                    pointer_chase_lam);
        RP_CALI_SUBKERNEL_END("POINTER_CHASE_1");

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  POINTER_CHASE : Unknown variant id = " << vid
                << std::endl;
    }

  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(
    POINTER_CHASE, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace basic
} // end namespace rajaperf
