//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES_NOVIEW.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void LTIMES_NOVIEW::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  LTIMES_NOVIEW_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("LTIMES_NOVIEW_1");
      #pragma omp target is_device_ptr(phidat, elldat, psidat) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(3)
      for (Index_type z = 0; z < num_z; ++z ) {
        for (Index_type g = 0; g < num_g; ++g ) {
          for (Index_type m = 0; m < num_m; ++m ) {
            for (Index_type d = 0; d < num_d; ++d ) {
              LTIMES_NOVIEW_BODY;
            }
          }
        }
      }
      RP_CALI_SUBKERNEL_END("LTIMES_NOVIEW_1");

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

      RP_CALI_SUBKERNEL_BEGIN("LTIMES_NOVIEW_1");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment(0, num_d),
                         RAJA::RangeSegment(0, num_z),
                         RAJA::RangeSegment(0, num_g),
                         RAJA::RangeSegment(0, num_m)),
        res,
        [=] (Index_type d, Index_type z, Index_type g, Index_type m) {
        LTIMES_NOVIEW_BODY;
      });
      RP_CALI_SUBKERNEL_END("LTIMES_NOVIEW_1");

    }
    stopTimer();

  } else {
     getCout() << "\n LTIMES_NOVIEW : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

void LTIMES_NOVIEW::defineOpenMPTargetVariantTunings()
{

  for (VariantID vid : {Base_OpenMPTarget, RAJA_OpenMPTarget}) {

    if (vid == RAJA_OpenMPTarget) {

      addVariantTuning<&LTIMES_NOVIEW::runOpenMPTargetVariant>(
          vid, "kernel");

    } else {

      addVariantTuning<&LTIMES_NOVIEW::runOpenMPTargetVariant>(
          vid, "default");

    }

  }

}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
