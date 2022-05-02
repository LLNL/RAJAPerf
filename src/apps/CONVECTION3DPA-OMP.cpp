//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives for loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "CONVECTION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

void CONVECTION3DPA::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  CONVECTION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_OpenMP: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#pragma omp parallel for
      for (int e = 0; e < NE; ++e) {


      } // element loop
    }
    stopTimer();

    break;
  }

  case RAJA_OpenMP: {

    // Currently Teams requires two policies if compiled with a device
    using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::omp_launch_t>;

    using outer_x = RAJA::expt::LoopPolicy<RAJA::omp_for_exec>;

    using inner_x = RAJA::expt::LoopPolicy<RAJA::loop_exec>;

    using inner_y = RAJA::expt::LoopPolicy<RAJA::loop_exec>;

    using inner_z = RAJA::expt::LoopPolicy<RAJA::loop_exec>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Grid is empty as the host does not need a compute grid to be specified
      RAJA::expt::launch<launch_policy>(
          RAJA::expt::Grid(),
          [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

          RAJA::expt::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {



            } // lambda (e)
          ); // RAJA::expt::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::expt::launch
    }  // loop over kernel reps
    stopTimer();

    return;
  }

  default:
    getCout() << "\n CONVECTION3DPA : Unknown OpenMP variant id = " << vid
              << std::endl;
  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace apps
} // end namespace rajaperf
