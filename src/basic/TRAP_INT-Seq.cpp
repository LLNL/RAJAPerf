//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"

#include "TRAP_INT-func.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void TRAP_INT::runSeqVariant(VariantID vid, size_t tune_idx)
{
#if !defined(RUN_RAJA_SEQ)
  RAJA_UNUSED_VAR(tune_idx);
#endif
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRAP_INT_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type sumx = m_sumx_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          TRAP_INT_BODY;
        }

        m_sumx += sumx * h;

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto trapint_base_lam = [=](Index_type i) -> Real_type {
                                Real_type x = x0 + i*h;
                                return trap_int_func(x, y, xp, yp);
                              };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type sumx = m_sumx_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          sumx += trapint_base_lam(i);
        }

        m_sumx += sumx * h;

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      if (tune_idx == 0) {

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          RAJA::ReduceSum<RAJA::seq_reduce, Real_type> sumx(m_sumx_init);

          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            TRAP_INT_BODY;
          });

          m_sumx += static_cast<Real_type>(sumx.get()) * h;

        }
        stopTimer();

      } else if (tune_idx == 1) {

        startTimer();
        for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          Real_type tsumx = m_sumx_init;

          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::plus>(&tsumx),
            [=] (Index_type i, Real_type& sumx) {
              TRAP_INT_BODY;
            }
          );

          m_sumx += static_cast<Real_type>(tsumx) * h;

        }
        stopTimer();

      } else {
        getCout() << "\n  TRAP_INT : Unknown Seq tuning index = " << tune_idx << std::endl;
      }

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  TRAP_INT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void TRAP_INT::setSeqTuningDefinitions(VariantID vid)
{
  addVariantTuningName(vid, "default");
  if (vid == RAJA_Seq) {
    addVariantTuningName(vid, "new");
  }
}

} // end namespace basic
} // end namespace rajaperf
