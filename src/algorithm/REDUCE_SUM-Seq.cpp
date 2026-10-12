//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

template < size_t tune_idx >
void REDUCE_SUM::runSeqVariant(VariantID vid)
{
#if !defined(RUN_RAJA_SEQ)
  RAJA_UNUSED_VAR(tune_idx);
#endif
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      if constexpr (tune_idx == 0) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("REDUCE_SUM_1");
          Real_type sum = m_sum_init;

          for (Index_type i = ibegin; i < iend; ++i ) {
            REDUCE_SUM_BODY;
          }

          m_sum = sum;
          RP_CALI_SUBKERNEL_END("REDUCE_SUM_1");

        }
        stopTimer();

      } else if constexpr (tune_idx == 1) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("REDUCE_SUM_1");
          RAJA::KahanSum<Real_type> sum(m_sum_init);

          for (Index_type i = ibegin; i < iend; ++i ) {
            REDUCE_SUM_BODY;
          }

          m_sum = sum.get();
          RP_CALI_SUBKERNEL_END("REDUCE_SUM_1");

        }
        stopTimer();

      } else if constexpr (tune_idx == 2) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("REDUCE_SUM_1");
          RAJA::BinaryTreeReduce<Real_type, RAJA::operators::plus<Real_type>> sum(m_sum_init);

          for (Index_type i = ibegin; i < iend; ++i ) {
            REDUCE_SUM_BODY;
          }

          m_sum = sum.get();
          RP_CALI_SUBKERNEL_END("REDUCE_SUM_1");

        }
        stopTimer();

      }

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto reduce_sum_base_lam = [=](Index_type i) {
                                 return x[i];
                               };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("REDUCE_SUM_1");
        Real_type sum = m_sum_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          sum += reduce_sum_base_lam(i);
        }

        m_sum = sum;
        RP_CALI_SUBKERNEL_END("REDUCE_SUM_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      if constexpr (tune_idx == 0) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("REDUCE_SUM_1");
          RAJA::ReduceSum<RAJA::seq_reduce, Real_type> sum(m_sum_init);

          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            [=](Index_type i) {
              REDUCE_SUM_BODY;
          });

          m_sum = sum.get();
          RP_CALI_SUBKERNEL_END("REDUCE_SUM_1");

        }
        stopTimer();

      } else if constexpr (tune_idx == 1) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("REDUCE_SUM_1");
          Real_type tsum = m_sum_init;

          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::plus>(&tsum),
            [=] (Index_type i,
              RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& sum) {
              REDUCE_SUM_BODY;
            }
          );

          m_sum = static_cast<Real_type>(tsum);
          RP_CALI_SUBKERNEL_END("REDUCE_SUM_1");

        }
        stopTimer();

      } else {
        getCout() << "\n  REDUCE_SUM : Unknown Seq tuning index = " << tune_idx << std::endl; 
      }

      break;
   }
#endif

    default : {
      getCout() << "\n  REDUCE_SUM : Unknown variant id = " << vid << std::endl;
    }

  }

}


void REDUCE_SUM::defineSeqVariantTunings()
{

  for (VariantID vid : {Base_Seq, Lambda_Seq, RAJA_Seq}) {

    addVariantTuning<&REDUCE_SUM::runSeqVariant<0>>(
        vid, "default");

    if (vid == Base_Seq) {

      addVariantTuning<&REDUCE_SUM::runSeqVariant<1>>(
          vid, "kahan", TuningAttribute::preferred_checksum);

      addVariantTuning<&REDUCE_SUM::runSeqVariant<2>>(
          vid, "cascade", TuningAttribute::preferred_checksum);

    }

    if (vid == RAJA_Seq) {

      addVariantTuning<&REDUCE_SUM::runSeqVariant<1>>(
          vid, "new");

    }

  }
}

} // end namespace algorithm
} // end namespace rajaperf
