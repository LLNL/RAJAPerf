//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST_3LOOP.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void INDEXLIST_3LOOP::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INDEXLIST_3LOOP_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      INDEXLIST_3LOOP_COUNTS_SETUP(DataSpace::Host);

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_3LOOP_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
        }
        RP_CALI_SUBKERNEL_END("INDEXLIST_3LOOP_1");

        Index_type count = 0;
        for (Index_type i = ibegin; i < iend+1; ++i ) {
          Index_type inc = counts[i];
          counts[i] = count;
          count += inc;
        }

        RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_3LOOP_2");
        for (Index_type i = ibegin; i < iend; ++i ) {
          INDEXLIST_3LOOP_MAKE_LIST;
        }
        RP_CALI_SUBKERNEL_END("INDEXLIST_3LOOP_2");

        m_len = counts[iend];

      }
      stopTimer();

      INDEXLIST_3LOOP_COUNTS_TEARDOWN(DataSpace::Host);

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      INDEXLIST_3LOOP_COUNTS_SETUP(DataSpace::Host);

      auto indexlist_conditional_lam = [=](Index_type i) {
                                  counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
                                };

      auto indexlist_make_list_lam = [=](Index_type i) {
                                  INDEXLIST_3LOOP_MAKE_LIST;
                                };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_3LOOP_1");
        for (Index_type i = ibegin; i < iend; ++i ) {
          indexlist_conditional_lam(i);
        }
        RP_CALI_SUBKERNEL_END("INDEXLIST_3LOOP_1");

        Index_type count = 0;
        for (Index_type i = ibegin; i < iend+1; ++i ) {
          Index_type inc = counts[i];
          counts[i] = count;
          count += inc;
        }

        RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_3LOOP_2");
        for (Index_type i = ibegin; i < iend; ++i ) {
          indexlist_make_list_lam(i);
        }
        RP_CALI_SUBKERNEL_END("INDEXLIST_3LOOP_2");

        m_len = counts[iend];

      }
      stopTimer();

      INDEXLIST_3LOOP_COUNTS_TEARDOWN(DataSpace::Host);

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      INDEXLIST_3LOOP_COUNTS_SETUP(DataSpace::Host);

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_3LOOP_1");
        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
          counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
        });
        RP_CALI_SUBKERNEL_END("INDEXLIST_3LOOP_1");
        RAJA::exclusive_scan_inplace<RAJA::seq_exec>( res,
            RAJA::make_span(counts+ibegin, iend+1-ibegin));

        RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_3LOOP_2");
        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
          if (counts[i] != counts[i+1]) {
            list[counts[i]] = i;
          }
        });
        RP_CALI_SUBKERNEL_END("INDEXLIST_3LOOP_2");

        m_len = counts[iend];

      }
      stopTimer();

      INDEXLIST_3LOOP_COUNTS_TEARDOWN(DataSpace::Host);

      break;
    }
#endif

    default : {
      getCout() << "\n  INDEXLIST_3LOOP : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(INDEXLIST_3LOOP, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace basic
} // end namespace rajaperf
