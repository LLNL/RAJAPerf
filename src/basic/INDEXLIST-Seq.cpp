//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void INDEXLIST::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INDEXLIST_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_1");
        Index_type count = 0;

        for (Index_type i = ibegin; i < iend; ++i ) {
          INDEXLIST_BODY;
        }

        m_len = count;
        RP_CALI_SUBKERNEL_END("INDEXLIST_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto indexlist_base_lam = [=](Index_type i, Index_type& count) {
                                 INDEXLIST_BODY
                               };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_1");
        Index_type count = 0;

        for (Index_type i = ibegin; i < iend; ++i ) {
          indexlist_base_lam(i, count);
        }

        m_len = count;
        RP_CALI_SUBKERNEL_END("INDEXLIST_1");

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  INDEXLIST : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(INDEXLIST, Seq, Base_Seq, Lambda_Seq)

} // end namespace basic
} // end namespace rajaperf
