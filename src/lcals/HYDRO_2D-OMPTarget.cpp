//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

void HYDRO_2D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  HYDRO_2D_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      #pragma omp target is_device_ptr(zadat, zbdat, zpdat, \
                                       zqdat, zrdat, zmdat) device( did )
      RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_1");
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      for (Index_type k = kbeg; k < kend; ++k ) {
        for (Index_type j = jbeg; j < jend; ++j ) {
          HYDRO_2D_BODY1;
        }
      }
      RP_CALI_SUBKERNEL_END("HYDRO_2D_1");

      #pragma omp target is_device_ptr(zudat, zvdat, zadat, \
                                       zbdat, zzdat, zrdat) device( did )
      RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_2");
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      for (Index_type k = kbeg; k < kend; ++k ) {
        for (Index_type j = jbeg; j < jend; ++j ) {
          HYDRO_2D_BODY2;
        }
      }
      RP_CALI_SUBKERNEL_END("HYDRO_2D_2");

      #pragma omp target is_device_ptr(zroutdat, zzoutdat, \
                                       zrdat, zudat, zzdat, zvdat) device( did )
      RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_3");
      #pragma omp teams distribute parallel for schedule(static, 1) collapse(2)
      for (Index_type k = kbeg; k < kend; ++k ) {
        for (Index_type j = jbeg; j < jend; ++j ) {
          HYDRO_2D_BODY3;
        }
      }
      RP_CALI_SUBKERNEL_END("HYDRO_2D_3");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    HYDRO_2D_VIEWS_RAJA;

    using EXECPOL =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>,
          RAJA::statement::Lambda<0>
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_1");
      RAJA::kernel_resource<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        res,
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY1_RAJA;
      });
      RP_CALI_SUBKERNEL_END("HYDRO_2D_1");

      RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_2");
      RAJA::kernel_resource<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        res,
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY2_RAJA;
      });
      RP_CALI_SUBKERNEL_END("HYDRO_2D_2");

      RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_3");
      RAJA::kernel_resource<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        res,
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY3_RAJA;
      });
      RP_CALI_SUBKERNEL_END("HYDRO_2D_3");

    }
    stopTimer();

  } else {
     getCout() << "\n  HYDRO_2D : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(HYDRO_2D, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
