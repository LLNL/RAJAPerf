//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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


void LTIMES::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();

  LTIMES_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(phidat, elldat, psidat) device( did )
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(3)
      for (Index_type z = 0; z < num_z; ++z ) {
        for (Index_type g = 0; g < num_g; ++g ) {
          for (Index_type m = 0; m < num_m; ++m ) {
            for (Index_type d = 0; d < num_d; ++d ) {
              LTIMES_BODY;
            }
          }
        }
      }

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    LTIMES_VIEWS_RANGES_RAJA;

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
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(IDRange(0, num_d),
                                               IZRange(0, num_z),
                                               IGRange(0, num_g),
                                               IMRange(0, num_m)),
        [=] (ID d, IZ z, IG g, IM m) {
        LTIMES_BODY_RAJA;
      });

    }
    stopTimer();

  } else {
     getCout() << "\n LTIMES : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
