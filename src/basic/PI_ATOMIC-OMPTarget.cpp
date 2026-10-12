//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;


void PI_ATOMIC::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_ATOMIC_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("PI_ATOMIC_1");
      initOpenMPDeviceData(pi, &m_pi_init, 1);

      #pragma omp target is_device_ptr(pi)
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        PI_ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_OMP);
      }

      getOpenMPDeviceData(&m_pi_final, pi, 1);
      m_pi_final *= 4.0;
      RP_CALI_SUBKERNEL_END("PI_ATOMIC_1");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("PI_ATOMIC_1");
      initOpenMPDeviceData(pi, &m_pi_init, 1);

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          PI_ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_RAJA_OMP);
      });

      getOpenMPDeviceData(&m_pi_final, pi, 1);
      m_pi_final *= 4.0;
      RP_CALI_SUBKERNEL_END("PI_ATOMIC_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  PI_ATOMIC : Unknown OMP Target variant id = " << vid << std::endl;
  }

  PI_ATOMIC_DATA_TEARDOWN;

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(PI_ATOMIC, OpenMPTarget, Base_OpenMPTarget, RAJA_OpenMPTarget)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
