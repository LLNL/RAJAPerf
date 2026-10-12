//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ZONAL_ACCUMULATION_3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;


void ZONAL_ACCUMULATION_3D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  ZONAL_ACCUMULATION_3D_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("ZONAL_ACCUMULATION_3D_1");
      #pragma omp target is_device_ptr(x0,x1,x2,x3,x4,x5,x6,x7, \
                                       vol, real_zones) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
        ZONAL_ACCUMULATION_3D_BODY_INDEX;
        ZONAL_ACCUMULATION_3D_BODY;
      }
      RP_CALI_SUBKERNEL_END("ZONAL_ACCUMULATION_3D_1");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    RAJA::TypedListSegment<Index_type> zones(real_zones, iend,
                                             res, RAJA::Unowned);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("ZONAL_ACCUMULATION_3D_1");
      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
        zones, [=](Index_type i) {
        ZONAL_ACCUMULATION_3D_BODY;
      });
      RP_CALI_SUBKERNEL_END("ZONAL_ACCUMULATION_3D_1");

    }
    stopTimer();

  } else {
    getCout() << "\n  ZONAL_ACCUMULATION_3D : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(ZONAL_ACCUMULATION_3D, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
