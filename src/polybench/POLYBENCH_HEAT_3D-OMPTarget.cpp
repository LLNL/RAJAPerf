//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_HEAT_3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

void POLYBENCH_HEAT_3D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_HEAT_3D_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_HEAT_3D_1");
      #pragma omp target is_device_ptr(A,B) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(3)
      for (Index_type i = 1; i < N-1; ++i ) {
        for (Index_type j = 1; j < N-1; ++j ) {
          for (Index_type k = 1; k < N-1; ++k ) {
            POLYBENCH_HEAT_3D_BODY1;
          }
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_HEAT_3D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_HEAT_3D_2");
      #pragma omp target is_device_ptr(A,B) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(3)
      for (Index_type i = 1; i < N-1; ++i ) {
        for (Index_type j = 1; j < N-1; ++j ) {
          for (Index_type k = 1; k < N-1; ++k ) {
            POLYBENCH_HEAT_3D_BODY2;
          }
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_HEAT_3D_2");

    }
    stopTimer();

  } else if (vid == RAJA_OpenMPTarget) {

    auto res{getOmpTargetResource()};

    POLYBENCH_HEAT_3D_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1, 2>,
          RAJA::statement::Lambda<0>
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_HEAT_3D_1");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                         RAJA::RangeSegment{1, N-1},
                         RAJA::RangeSegment{1, N-1}),
        res,
        [=] (Index_type i, Index_type j, Index_type k) {
          POLYBENCH_HEAT_3D_BODY1_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_HEAT_3D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_HEAT_3D_2");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                         RAJA::RangeSegment{1, N-1},
                         RAJA::RangeSegment{1, N-1}),
        res,
        [=] (Index_type i, Index_type j, Index_type k) {
          POLYBENCH_HEAT_3D_BODY2_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_HEAT_3D_2");

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_HEAT_3D : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_HEAT_3D, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
