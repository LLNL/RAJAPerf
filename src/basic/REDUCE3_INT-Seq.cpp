//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#include <limits>
#include <iostream>

namespace rajaperf
{
namespace basic
{


void REDUCE3_INT::runSeqVariant(VariantID vid, size_t tune_idx)
{
#if !defined(RUN_RAJA_SEQ)
  RAJA_UNUSED_VAR(tune_idx);
#endif
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE3_INT_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Int_type vsum = m_vsum_init;
        Int_type vmin = m_vmin_init;
        Int_type vmax = m_vmax_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          REDUCE3_INT_BODY;
        }

        m_vsum += vsum;
        m_vmin = RAJA_MIN(m_vmin, vmin);
        m_vmax = RAJA_MAX(m_vmax, vmax);

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto reduce3_base_lam = [=](Index_type i) -> Int_type {
                                return vec[i];
                              };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Int_type vsum = m_vsum_init;
        Int_type vmin = m_vmin_init;
        Int_type vmax = m_vmax_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          vsum += reduce3_base_lam(i);
          vmin = RAJA_MIN(vmin, reduce3_base_lam(i));
          vmax = RAJA_MAX(vmax, reduce3_base_lam(i));
        }

        m_vsum += vsum;
        m_vmin = RAJA_MIN(m_vmin, vmin);
        m_vmax = RAJA_MAX(m_vmax, vmax);

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      if (tune_idx == 0) {

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          RAJA::ReduceSum<RAJA::seq_reduce, Int_type> vsum(m_vsum_init);
          RAJA::ReduceMin<RAJA::seq_reduce, Int_type> vmin(m_vmin_init);
          RAJA::ReduceMax<RAJA::seq_reduce, Int_type> vmax(m_vmax_init);
  
          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            REDUCE3_INT_BODY_RAJA;
          });
  
          m_vsum += static_cast<Int_type>(vsum.get());
          m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
          m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));
  
        }
        stopTimer();

      } else if (tune_idx == 1) {

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          Int_type tvsum = m_vsum_init; 
          Int_type tvmin = m_vmin_init; 
          Int_type tvmax = m_vmax_init; 

          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::plus>(&tvsum),
            RAJA::expt::Reduce<RAJA::operators::minimum>(&tvmin),
            RAJA::expt::Reduce<RAJA::operators::maximum>(&tvmax),
            [=](Index_type i, Int_type& vsum, Int_type& vmin, Int_type& vmax) {
              REDUCE3_INT_BODY;
            }
          );

          m_vsum += static_cast<Int_type>(tvsum);
          m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(tvmin));
          m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(tvmax));

        }
        stopTimer();

      } else {
        getCout() << "\n  REDUCE3_INT : Unknown Seq tuning index = " << tune_idx << std::endl;
      }

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  REDUCE3_INT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void REDUCE3_INT::setSeqTuningDefinitions(VariantID vid)
{
  addVariantTuningName(vid, "default");
  if (vid == RAJA_Seq) {
    addVariantTuningName(vid, "new");
  }
}

} // end namespace basic
} // end namespace rajaperf
