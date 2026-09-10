//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

using namespace ltimes_idx;

void LTIMES::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  LTIMES_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
      #pragma omp target firstprivate(psi, ell, phi, num_z, num_g, num_m, num_d) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(3)
      for (RAJA::Index_type iz = 0; iz < *num_z; ++iz ) {
        for (RAJA::Index_type ig = 0; ig < *num_g; ++ig ) {
          for (RAJA::Index_type im = 0; im < *num_m; ++im ) {
            IZ z(iz);
            IG g(ig);
            IM m(im);
            for (ID d(0); d < num_d; ++d ) {
              LTIMES_BODY;
            }
          }
        }
      }
      RP_CALI_SUBKERNEL_END("LTIMES_1");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<1, 2, 3>, // z, g, m
          RAJA::statement::For<0, RAJA::seq_exec,         // d
            RAJA::statement::Lambda<0>
          >
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
      RAJA::kernel_resource<EXEC_POL>( RAJA::make_tuple(IDRange(0, *num_d),
                                                        IZRange(0, *num_z),
                                                        IGRange(0, *num_g),
                                                        IMRange(0, *num_m)),
        res,
        [=] (ID d, IZ z, IG g, IM m) {
        LTIMES_BODY;
      });
      RP_CALI_SUBKERNEL_END("LTIMES_1");

    }
    stopTimer();

  } else {
     getCout() << "\n LTIMES : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

void LTIMES::defineOpenMPTargetVariantTunings()
{

  for (VariantID vid : {Base_OpenMPTarget, RAJA_OpenMPTarget}) {

    if (vid == RAJA_OpenMPTarget) {

      addVariantTuning<&LTIMES::runOpenMPTargetVariant>(
        vid, "kernel");

    } else {

      addVariantTuning<&LTIMES::runOpenMPTargetVariant>(
        vid, "default");

    }

  }

}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
