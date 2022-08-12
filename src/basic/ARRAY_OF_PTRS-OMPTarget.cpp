//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ARRAY_OF_PTRS.hpp"

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

#define ARRAY_OF_PTRS_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(x_data, m_x, array_size*iend, did, hid); \
  allocAndInitOpenMPDeviceData(y, m_y, iend, did, hid); \
  ARRAY_OF_PTRS_DATA_SETUP_X_ARRAY

#define ARRAY_OF_PTRS_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_y, y, iend, hid, did); \
  deallocOpenMPDeviceData(x_data, did); \
  deallocOpenMPDeviceData(y, did);


void ARRAY_OF_PTRS::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ARRAY_OF_PTRS_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    ARRAY_OF_PTRS_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(y) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ARRAY_OF_PTRS_BODY(x);
      }

    }
    stopTimer();

    ARRAY_OF_PTRS_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    ARRAY_OF_PTRS_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        ARRAY_OF_PTRS_BODY(x);
      });

    }
    stopTimer();

    ARRAY_OF_PTRS_DATA_TEARDOWN_OMP_TARGET;

  } else {
     getCout() << "\n  ARRAY_OF_PTRS : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP