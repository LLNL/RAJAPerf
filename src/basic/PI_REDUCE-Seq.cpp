//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
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


void PI_REDUCE::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
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

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::seq_reduce, Real_type> pi(m_pi_init);

        RAJA::forall<RAJA::loop_exec>( RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
            PI_REDUCE_BODY;
        });

        m_pi = 4.0 * pi.get();

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  PI_REDUCE : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace basic
} // end namespace rajaperf
