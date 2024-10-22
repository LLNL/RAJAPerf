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

#define INDEXLIST_3LOOP_DATA_SETUP_Seq \
  Index_type* counts = new Index_type[iend+1];

#define INDEXLIST_3LOOP_DATA_TEARDOWN_Seq \
  delete[] counts; counts = nullptr;



void INDEXLIST_3LOOP::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INDEXLIST_3LOOP_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      INDEXLIST_3LOOP_DATA_SETUP_Seq;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
        }

        Index_type count = 0;

        for (Index_type i = ibegin; i < iend+1; ++i ) {
          Index_type inc = counts[i];
          counts[i] = count;
          count += inc;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          INDEXLIST_3LOOP_MAKE_LIST;
        }

        m_len = counts[iend];

      }
      stopTimer();

      INDEXLIST_3LOOP_DATA_TEARDOWN_Seq;

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      INDEXLIST_3LOOP_DATA_SETUP_Seq;

      auto indexlist_conditional_lam = [=](Index_type i) {
                                  counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
                                };

      auto indexlist_make_list_lam = [=](Index_type i) {
                                  INDEXLIST_3LOOP_MAKE_LIST;
                                };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          indexlist_conditional_lam(i);
        }

        Index_type count = 0;

        for (Index_type i = ibegin; i < iend+1; ++i ) {
          Index_type inc = counts[i];
          counts[i] = count;
          count += inc;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          indexlist_make_list_lam(i);
        }

        m_len = counts[iend];

      }
      stopTimer();

      INDEXLIST_3LOOP_DATA_TEARDOWN_Seq;

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      INDEXLIST_3LOOP_DATA_SETUP_Seq;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
          counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
        });

        RAJA::exclusive_scan_inplace<RAJA::seq_exec>( res,
            RAJA::make_span(counts+ibegin, iend+1-ibegin));

        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
          if (counts[i] != counts[i+1]) {
            list[counts[i]] = i;
          }
        });

        m_len = counts[iend];

      }
      stopTimer();

      INDEXLIST_3LOOP_DATA_TEARDOWN_Seq;

      break;
    }
#endif

    default : {
      getCout() << "\n  INDEXLIST_3LOOP : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace basic
} // end namespace rajaperf
