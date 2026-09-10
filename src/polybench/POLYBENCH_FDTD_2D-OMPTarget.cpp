//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FDTD_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

void POLYBENCH_FDTD_2D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_FDTD_2D_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_1");
      #pragma omp target is_device_ptr(ey,fict) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type j = 0; j < ny; j++) {
        POLYBENCH_FDTD_2D_BODY1;
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_2");
      #pragma omp target is_device_ptr(ey,hz) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      for (Index_type i = 1; i < nx; i++) {
        for (Index_type j = 0; j < ny; j++) {
          POLYBENCH_FDTD_2D_BODY2;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_3");
      #pragma omp target is_device_ptr(ex,hz) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      for (Index_type i = 0; i < nx; i++) {
        for (Index_type j = 1; j < ny; j++) {
          POLYBENCH_FDTD_2D_BODY3;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_3");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_4");
      #pragma omp target is_device_ptr(ex,ey,hz) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      for (Index_type i = 0; i < nx - 1; i++) {
        for (Index_type j = 0; j < ny - 1; j++) {
          POLYBENCH_FDTD_2D_BODY4;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_4");

      t = (t+1) % m_tsteps;
    }
    stopTimer();

  } else if (vid == RAJA_OpenMPTarget) {

    auto res{getOmpTargetResource()};

    POLYBENCH_FDTD_2D_VIEWS_RAJA;

    using EXEC_POL1 = RAJA::omp_target_parallel_for_exec<threads_per_team>;

    using EXEC_POL234 =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>,
          RAJA::statement::Lambda<0>
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_1");
      RAJA::forall<EXEC_POL1>( res, RAJA::RangeSegment(0, ny),
       [=] (Index_type j) {
         POLYBENCH_FDTD_2D_BODY1_RAJA;
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_2");
      RAJA::kernel_resource<EXEC_POL234>(
        RAJA::make_tuple(RAJA::RangeSegment{1, nx},
                         RAJA::RangeSegment{0, ny}),
        res,
        [=] (Index_type i, Index_type j) {
          POLYBENCH_FDTD_2D_BODY2_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_3");
      RAJA::kernel_resource<EXEC_POL234>(
        RAJA::make_tuple(RAJA::RangeSegment{0, nx},
                         RAJA::RangeSegment{1, ny}),
        res,
        [=] (Index_type i, Index_type j) {
          POLYBENCH_FDTD_2D_BODY3_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_3");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_4");
      RAJA::kernel_resource<EXEC_POL234>(
        RAJA::make_tuple(RAJA::RangeSegment{0, nx-1},
                         RAJA::RangeSegment{0, ny-1}),
        res,
        [=] (Index_type i, Index_type j) {
          POLYBENCH_FDTD_2D_BODY4_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_4");

      t = (t+1) % m_tsteps;
    } // run_reps
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_FDTD_2D : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_FDTD_2D, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

