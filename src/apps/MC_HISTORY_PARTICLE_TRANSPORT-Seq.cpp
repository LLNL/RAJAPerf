//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MC_HISTORY_PARTICLE_TRANSPORT.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void MC_HISTORY_PARTICLE_TRANSPORT::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = 1;
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MC_HISTORY_PARTICLE_TRANSPORT_DATA_SETUP(vid, iend);

#if defined(RUN_RAJA_SEQ)
  auto MC_HISTORY_PARTICLE_TRANSPORT_lam = [&](Index_type i) {
                             MC_HISTORY_PARTICLE_TRANSPORT_BODY
                           };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      //MC_HISTORY_PARTICLE_TRANSPORT_DATA_SETUP(vid, iend);
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
        MC_HISTORY_PARTICLE_TRANSPORT_RESET
        for (Index_type i = ibegin; i < iend; ++i ) {
          MC_HISTORY_PARTICLE_TRANSPORT_BODY
        }
      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
        MC_HISTORY_PARTICLE_TRANSPORT_RESET
        for (Index_type i = ibegin; i < iend; ++i ) {
          MC_HISTORY_PARTICLE_TRANSPORT_lam(i);
        }

      }
      stopTimer();
      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
        MC_HISTORY_PARTICLE_TRANSPORT_RESET
        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend), MC_HISTORY_PARTICLE_TRANSPORT_lam);

      }
      stopTimer();
      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  MC_HISTORY_PARTICLE_TRANSPORT : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(MC_HISTORY_PARTICLE_TRANSPORT, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace apps
} // end namespace rajaperf
