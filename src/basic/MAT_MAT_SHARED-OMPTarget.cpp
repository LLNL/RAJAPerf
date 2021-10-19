//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT_SHARED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {


  void MAT_MAT_SHARED::runOpenMPTargetVariant(VariantID vid) {
    const Index_type run_reps = getRunReps();

    switch (vid) {

    default: {

      std::cout << "\n MAT_MAT_SHARED : Unknown OpenMPTarget variant id = " << vid << std::endl;
      break;
    }
    }
  }

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_TARGET_OPENMP
