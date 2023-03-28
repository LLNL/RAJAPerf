//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MEMCPY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

#define MEMCPY_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(x, m_x, iend, did, hid); \
  allocAndInitOpenMPDeviceData(y, m_y, iend, did, hid);

#define MEMCPY_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_y, y, iend, hid, did); \
  deallocOpenMPDeviceData(x, did); \
  deallocOpenMPDeviceData(y, did);


void MEMCPY::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMCPY_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    MEMCPY_DATA_SETUP_OMP_TARGET

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(x, y) device( did )
      #pragma omp teams distribute parallel for \
              thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        MEMCPY_BODY;
      }

    }
    stopTimer();

    MEMCPY_DATA_TEARDOWN_OMP_TARGET

  } else if ( vid == RAJA_OpenMPTarget ) {

    MEMCPY_DATA_SETUP_OMP_TARGET

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
        RAJA::RangeSegment(ibegin, iend),
        [=](Index_type i) {
          MEMCPY_BODY;
      });

    }
    stopTimer();

    MEMCPY_DATA_TEARDOWN_OMP_TARGET

  } else {
    getCout() << "\n  MEMCPY : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
