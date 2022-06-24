//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "BLOCK_DIAG_MAT_VEC_REORDER.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {


  void BLOCK_DIAG_MAT_VEC_REORDER::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
    const Index_type run_reps = getRunReps();

    switch (vid) {

    default: {

      getCout() << "\n BLOCK_DIAG_MAT_VEC_REORDER : Unknown OpenMPTarget variant id = " << vid << std::endl;
      break;
    }
    }
  }

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_TARGET_OPENMP
