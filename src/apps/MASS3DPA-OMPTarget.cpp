//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {


void MASS3DPA::runOpenMPTargetVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();

  switch (vid) {

  default: {

    getCout() << "\n MASS3DPA : Unknown OpenMPTarget variant id = " << vid << std::endl;
    break;
  }
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_TARGET_OPENMP
