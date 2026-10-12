//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;


void ENERGY::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ENERGY_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("ENERGY_1");
      #pragma omp target is_device_ptr(e_new, e_old, delvc, \
                                       p_old, q_old, work) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ENERGY_BODY1;
      }
      RP_CALI_SUBKERNEL_END("ENERGY_1");

      RP_CALI_SUBKERNEL_BEGIN("ENERGY_2");
      #pragma omp target is_device_ptr(delvc, q_new, compHalfStep, \
                                       pHalfStep, e_new, bvc, pbvc, \
                                       ql_old, qq_old) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ENERGY_BODY2;
      }
      RP_CALI_SUBKERNEL_END("ENERGY_2");

      RP_CALI_SUBKERNEL_BEGIN("ENERGY_3");
      #pragma omp target is_device_ptr(e_new, delvc, p_old, \
                                       q_old, pHalfStep, q_new) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ENERGY_BODY3;
      }
      RP_CALI_SUBKERNEL_END("ENERGY_3");

      RP_CALI_SUBKERNEL_BEGIN("ENERGY_4");
      #pragma omp target is_device_ptr(e_new, work) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ENERGY_BODY4;
      }
      RP_CALI_SUBKERNEL_END("ENERGY_4");

      RP_CALI_SUBKERNEL_BEGIN("ENERGY_5");
      #pragma omp target is_device_ptr(delvc, pbvc, e_new, vnewc, \
                                       bvc, p_new, ql_old, qq_old, \
                                       p_old, q_old, pHalfStep, q_new) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ENERGY_BODY5;
      }
      RP_CALI_SUBKERNEL_END("ENERGY_5");

      RP_CALI_SUBKERNEL_BEGIN("ENERGY_6");
      #pragma omp target is_device_ptr(delvc, pbvc, e_new, vnewc, \
                                       bvc, p_new, q_new, ql_old, qq_old) \
                                       device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ENERGY_BODY6;
      }
      RP_CALI_SUBKERNEL_END("ENERGY_6");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_1");
        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ENERGY_BODY1;
        });
        RP_CALI_SUBKERNEL_END("ENERGY_1");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_2");
        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ENERGY_BODY2;
        });
        RP_CALI_SUBKERNEL_END("ENERGY_2");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_3");
        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ENERGY_BODY3;
        });
        RP_CALI_SUBKERNEL_END("ENERGY_3");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_4");
        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ENERGY_BODY4;
        });
        RP_CALI_SUBKERNEL_END("ENERGY_4");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_5");
        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ENERGY_BODY5;
        });
        RP_CALI_SUBKERNEL_END("ENERGY_5");

        RP_CALI_SUBKERNEL_BEGIN("ENERGY_6");
        RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ENERGY_BODY6;
        });
        RP_CALI_SUBKERNEL_END("ENERGY_6");

      }); // end sequential region (for single-source code)

    }
    stopTimer();

  } else {
     getCout() << "\n  ENERGY : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(ENERGY, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
