//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void DAXPY_ATOMIC::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DAXPY_ATOMIC_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DAXPY_ATOMIC_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          DAXPY_ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_SEQ);
        }
        RP_CALI_SUBKERNEL_END("DAXPY_ATOMIC_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto daxpy_atomic_lam = [=](Index_type i) {
            DAXPY_ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_SEQ);
          };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DAXPY_ATOMIC_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          daxpy_atomic_lam(i);
        }
        RP_CALI_SUBKERNEL_END("DAXPY_ATOMIC_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DAXPY_ATOMIC_1");
        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
            DAXPY_ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_RAJA_SEQ);
        });
        RP_CALI_SUBKERNEL_END("DAXPY_ATOMIC_1");

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  DAXPY_ATOMIC : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(DAXPY_ATOMIC, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace basic
} // end namespace rajaperf
