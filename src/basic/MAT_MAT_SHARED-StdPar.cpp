//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT_SHARED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_STDPAR)


#include <iostream>

namespace rajaperf {
namespace basic {

void MAT_MAT_SHARED::runStdParVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{

  switch (vid) {
  default: {
    getCout() << "\n  MAT_MAT_SHARED : Unknown variant id = " << vid
              << std::endl;
  }
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_STDPAR

