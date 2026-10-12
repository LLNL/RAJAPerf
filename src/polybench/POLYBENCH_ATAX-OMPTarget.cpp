//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ATAX.hpp"

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

void POLYBENCH_ATAX::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_ATAX_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ATAX_1");
      #pragma omp target is_device_ptr(x,y,tmp,A) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = 0; i < N; ++i ) {
        POLYBENCH_ATAX_BODY1;
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_ATAX_BODY2;
        }
        POLYBENCH_ATAX_BODY3;
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_ATAX_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ATAX_2");
      #pragma omp target is_device_ptr(y,tmp,A) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type j = 0; j < N; ++j ) {
        POLYBENCH_ATAX_BODY4;
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_ATAX_BODY5;
        }
        POLYBENCH_ATAX_BODY6;
      }
      RP_CALI_SUBKERNEL_END("POLYBENCH_ATAX_2");

    }
    stopTimer();

  } else if (vid == RAJA_OpenMPTarget) {

    auto res{getOmpTargetResource()};

    POLYBENCH_ATAX_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<threads_per_team>,
          RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
          RAJA::statement::For<1, RAJA::seq_exec,
            RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
          >,
          RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
        >
      >;

    using EXEC_POL2 =
      RAJA::KernelPolicy<
        RAJA::statement::For<1, RAJA::omp_target_parallel_for_exec<threads_per_team>,
          RAJA::statement::Lambda<0, RAJA::Segs<1>, RAJA::Params<0>>,
          RAJA::statement::For<0, RAJA::seq_exec,
            RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
          >,
          RAJA::statement::Lambda<2, RAJA::Segs<1>, RAJA::Params<0>>
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ATAX_1");
      RAJA::kernel_param_resource<EXEC_POL1>(
        RAJA::make_tuple(RAJA::RangeSegment{0, N},
                         RAJA::RangeSegment{0, N}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] (Index_type i, Real_type &dot) {
          POLYBENCH_ATAX_BODY1_RAJA;
        },
        [=] (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_ATAX_BODY2_RAJA;
        },
        [=] (Index_type i, Real_type &dot) {
          POLYBENCH_ATAX_BODY3_RAJA;
        }

      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_ATAX_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ATAX_2");
      RAJA::kernel_param_resource<EXEC_POL2>(
        RAJA::make_tuple(RAJA::RangeSegment{0, N},
                         RAJA::RangeSegment{0, N}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] (Index_type j, Real_type &dot) {
          POLYBENCH_ATAX_BODY4_RAJA;
        },
        [=] (Index_type i, Index_type j , Real_type &dot) {
          POLYBENCH_ATAX_BODY5_RAJA;
        },
        [=] (Index_type j, Real_type &dot) {
          POLYBENCH_ATAX_BODY6_RAJA;
        }

      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_ATAX_2");

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_ATAX : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_ATAX, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

