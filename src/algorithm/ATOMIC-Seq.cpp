//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{


template < size_t replication >
void ATOMIC::runSeqVariantReplicate(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ATOMIC_DATA_SETUP(replication);

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("ATOMIC_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_SEQ, i, ATOMIC_VALUE);
        }
        RP_CALI_SUBKERNEL_END("ATOMIC_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto atomic_base_lam = [=](Index_type i) {
                                 ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_SEQ, i, ATOMIC_VALUE);
                               };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
     for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("ATOMIC_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          atomic_base_lam(i);
        }
        RP_CALI_SUBKERNEL_END("ATOMIC_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("ATOMIC_1");
        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
            ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_RAJA_SEQ, i, ATOMIC_VALUE);
        });
        RP_CALI_SUBKERNEL_END("ATOMIC_1");

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  ATOMIC : Unknown variant id = " << vid << std::endl;
    }

  }

  ATOMIC_DATA_TEARDOWN(replication);

}


void ATOMIC::defineSeqVariantTunings()
{
  for (VariantID vid : {Base_Seq, Lambda_Seq, RAJA_Seq}) {

    seq_for(cpu_atomic_replications_type{}, [&](auto replication) {

      if (run_params.numValidAtomicReplication() == 0u ||
          run_params.validAtomicReplication(replication)) {

        addVariantTuning<&ATOMIC::runSeqVariantReplicate<replication>>(
            vid, "replicate_"+std::to_string(replication));

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf
