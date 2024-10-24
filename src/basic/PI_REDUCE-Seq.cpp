//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void PI_REDUCE::runSeqVariant(VariantID vid, size_t tune_idx)
{
#if !defined(RUN_RAJA_SEQ)
  RAJA_UNUSED_VAR(tune_idx);
#endif
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_REDUCE_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type pi = m_pi_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          PI_REDUCE_BODY;
        }

        m_pi = 4.0 * pi;

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto pireduce_base_lam = [=](Index_type i) -> Real_type {
                                 double x = (double(i) + 0.5) * dx;
                                 return dx / (1.0 + x * x);
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type pi = m_pi_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          pi += pireduce_base_lam(i);
        }

        m_pi = 4.0 * pi;

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      RAJA::resources::Host res;

      if (tune_idx == 0) {

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          RAJA::ReduceSum<RAJA::seq_reduce, Real_type> pi(m_pi_init);
  
          RAJA::forall<RAJA::seq_exec>(res, RAJA::RangeSegment(ibegin, iend),
            [=](Index_type i) {
              PI_REDUCE_BODY;
          });

          m_pi = 4.0 * pi.get();

        }
        stopTimer();

      } else if (tune_idx == 1) {

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          Real_type tpi = m_pi_init;
 
          RAJA::forall<RAJA::seq_exec>(res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::plus>(&tpi),
            [=] (Index_type i,
              RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& pi) {
              PI_REDUCE_BODY;
            }
          );

          m_pi = static_cast<Real_type>(tpi) * 4.0;

        }
        stopTimer();       
  
      } else {
        getCout() << "\n  PI_REDUCE : Unknown Seq tuning index = " << tune_idx << std::endl;
      }

      break;
    }
#endif

    default : {
      getCout() << "\n  PI_REDUCE : Unknown variant id = " << vid << std::endl;
    }

  }

}

void PI_REDUCE::setSeqTuningDefinitions(VariantID vid)
{
  addVariantTuningName(vid, "default");
  if (vid == RAJA_Seq) {
    addVariantTuningName(vid, "new");
  }
}

} // end namespace basic
} // end namespace rajaperf
