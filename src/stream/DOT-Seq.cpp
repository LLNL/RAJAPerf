//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DOT.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

template < size_t tune_idx >
void DOT::runSeqVariant(VariantID vid)
{
#if !defined(RUN_RAJA_SEQ)
  RAJA_UNUSED_VAR(tune_idx);
#endif
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DOT_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      if constexpr (tune_idx == 0) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("DOT_1");
          Real_type dot = m_dot_init;

          for (Index_type i = ibegin; i < iend; ++i ) {
            DOT_BODY;
          }

          m_dot += dot;
          RP_CALI_SUBKERNEL_END("DOT_1");

        }
        stopTimer();

      } else if constexpr (tune_idx == 1) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("DOT_1");
          RAJA::KahanSum<Real_type> dot(m_dot_init);

          for (Index_type i = ibegin; i < iend; ++i ) {
            DOT_BODY;
          }

          m_dot += dot.get();
          RP_CALI_SUBKERNEL_END("DOT_1");

        }
        stopTimer();

      } else if constexpr (tune_idx == 2) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("DOT_1");
          RAJA::BinaryTreeReduce<Real_type, RAJA::operators::plus<Real_type>> dot(m_dot_init);

          for (Index_type i = ibegin; i < iend; ++i ) {
            DOT_BODY;
          }

          m_dot += dot.get();
          RP_CALI_SUBKERNEL_END("DOT_1");

        }
        stopTimer();

      }

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto dot_base_lam = [=](Index_type i) -> Real_type {
                            return a[i] * b[i];
                          };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DOT_1");
        Real_type dot = m_dot_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          dot += dot_base_lam(i);
        }

        m_dot += dot;
        RP_CALI_SUBKERNEL_END("DOT_1");

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

          RP_CALI_SUBKERNEL_BEGIN("DOT_1");
          RAJA::ReduceSum<RAJA::seq_reduce, Real_type> dot(m_dot_init);
  
          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            DOT_BODY;
          });

          m_dot += static_cast<Real_type>(dot.get());
          RP_CALI_SUBKERNEL_END("DOT_1");

        }
        stopTimer();

      } else if constexpr (tune_idx == 1) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("DOT_1");
          Real_type tdot = m_dot_init;

          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::plus>(&tdot),
            [=] (Index_type i,
              RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& dot) {
              DOT_BODY;
            }
          );

          m_dot += static_cast<Real_type>(tdot);
          RP_CALI_SUBKERNEL_END("DOT_1");

        }
        stopTimer();

      } else {
        getCout() << "\n  DOT : Unknown Seq tuning index = " << tune_idx << std::endl;
      }

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  DOT : Unknown variant id = " << vid << std::endl;
    }

  }

}


void DOT::defineSeqVariantTunings()
{

  for (VariantID vid : {Base_Seq, Lambda_Seq, RAJA_Seq}) {

    addVariantTuning<&DOT::runSeqVariant<0>>(
        vid, "default");

    if (vid == Base_Seq) {

      addVariantTuning<&DOT::runSeqVariant<1>>(
          vid, "kahan", TuningAttribute::preferred_checksum);

      addVariantTuning<&DOT::runSeqVariant<2>>(
          vid, "cascade", TuningAttribute::preferred_checksum);

    }

    if (vid == RAJA_Seq) {

      addVariantTuning<&DOT::runSeqVariant<1>>(
          vid, "new");

    }

  }

}

} // end namespace stream
} // end namespace rajaperf
