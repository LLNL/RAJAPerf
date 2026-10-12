//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMVER.hpp"

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

void POLYBENCH_GEMVER::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_GEMVER_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_1");
      #pragma omp target is_device_ptr(A,u1,v1,u2,v2) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      for (Index_type i = 0; i < n; i++) {
        for(Index_type j = 0; j < n; j++) {
          POLYBENCH_GEMVER_BODY1;
        }
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_2");
      #pragma omp target is_device_ptr(A,x,y) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = 0; i < n; i++) {
        POLYBENCH_GEMVER_BODY2;
        for (Index_type j = 0; j < n; j++) {
          POLYBENCH_GEMVER_BODY3;
        }
        POLYBENCH_GEMVER_BODY4;
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_3");
      #pragma omp target is_device_ptr(x,z) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = 0; i < n; i++) {
        POLYBENCH_GEMVER_BODY5;
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_3");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_4");
      #pragma omp target is_device_ptr(A,w,x) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = 0; i < n; i++) {
        POLYBENCH_GEMVER_BODY6;
        for (Index_type j = 0; j < n; j++) {
          POLYBENCH_GEMVER_BODY7;
        }
        POLYBENCH_GEMVER_BODY8;
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_4");

    } // end run_reps
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()}; 

    POLYBENCH_GEMVER_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>,
          RAJA::statement::Lambda<0>
        >
      >;

    using EXEC_POL24 =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<threads_per_team>,
          RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
          RAJA::statement::For<1, RAJA::seq_exec,
            RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
          >,
          RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
        >
      >;

    using EXEC_POL3 = RAJA::omp_target_parallel_for_exec<threads_per_team>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_1");
      RAJA::kernel_resource<EXEC_POL1>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        res,
        [=] (Index_type i, Index_type j) {
          POLYBENCH_GEMVER_BODY1_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_2");
      RAJA::kernel_param_resource<EXEC_POL24>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] (Index_type /* i */, Real_type &dot) {
          POLYBENCH_GEMVER_BODY2_RAJA;
        },
        [=] (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_GEMVER_BODY3_RAJA;
        },
        [=] (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY4_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_3");
      RAJA::forall<EXEC_POL3> (res, RAJA::RangeSegment{0, n},
        [=] (Index_type i) {
          POLYBENCH_GEMVER_BODY5_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_3");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_4");
      RAJA::kernel_param_resource<EXEC_POL24>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY6_RAJA;
        },
        [=] (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_GEMVER_BODY7_RAJA;
        },
        [=] (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY8_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_4");

    }
    stopTimer();

  } else {
     getCout() << "\n  POLYBENCH_GEMVER : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_GEMVER, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

