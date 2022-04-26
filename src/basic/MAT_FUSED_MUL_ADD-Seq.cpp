//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_FUSED_MUL_ADD.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

void MAT_FUSED_MUL_ADD::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;
  const Index_type Ne = m_Ne;

  MAT_FUSED_MUL_ADD_DATA_SETUP;

  switch (vid) {

  case Base_Seq: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    } // number of iterations
    stopTimer();

    break;
  }

#if defined(RUN_RAJA_SEQ)
  case Lambda_Seq: {


    startTimer();
    for (Index_type irep = 0; irep < run_reps; ++irep) {
    } // irep
    stopTimer();

    break;
  }

  case RAJA_Seq: {
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }  // loop over kernel reps
    stopTimer();

    break;
  }
#endif // RUN_RAJA_SEQ

  default: {
    getCout() << "\n  MAT_FUSED_MUL_ADD : Unknown variant id = " << vid
              << std::endl;
  }
  }
}

} // end namespace basic
} // end namespace rajaperf
