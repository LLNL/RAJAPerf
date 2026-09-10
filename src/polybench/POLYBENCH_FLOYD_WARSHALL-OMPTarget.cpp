//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

void POLYBENCH_FLOYD_WARSHALL::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_FLOYD_WARSHALL_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      for (Index_type k = 0; k < N; ++k) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FLOYD_WARSHALL_k");
        #pragma omp target is_device_ptr(pout,pin) device( did )
        #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
        for (Index_type i = 0; i < N; ++i) {
          for (Index_type j = 0; j < N; ++j) {
            POLYBENCH_FLOYD_WARSHALL_BODY;
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_FLOYD_WARSHALL_k");

      }

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    POLYBENCH_FLOYD_WARSHALL_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                    RAJA::ArgList<1, 2>,
            RAJA::statement::Lambda<0>
          >
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FLOYD_WARSHALL_k");
      RAJA::kernel_resource<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                                        RAJA::RangeSegment{0, N},
                                                        RAJA::RangeSegment{0, N}),
        res,
        [=] (Index_type k, Index_type i, Index_type j) {
          POLYBENCH_FLOYD_WARSHALL_BODY_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_FLOYD_WARSHALL_k");

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_FLOYD_WARSHALL, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

