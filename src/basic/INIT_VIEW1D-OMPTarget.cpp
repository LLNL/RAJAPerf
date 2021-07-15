//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D.hpp"

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

#define INIT_VIEW1D_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  allocAndInitOpenMPDeviceData(a, m_a, getActualProblemSize(), did, hid);

#define INIT_VIEW1D_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_a, a, getActualProblemSize(), hid, did); \
  deallocOpenMPDeviceData(a, did);


void INIT_VIEW1D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INIT_VIEW1D_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    INIT_VIEW1D_DATA_SETUP_OMP_TARGET;                 

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(a) device( did )
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) 
      for (Index_type i = ibegin; i < iend; ++i ) {
        INIT_VIEW1D_BODY;
      }

    }
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

     INIT_VIEW1D_DATA_SETUP_OMP_TARGET;

     INIT_VIEW1D_VIEW_RAJA;

     startTimer();
     for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       //RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
       //  RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
       //  INIT_VIEW1D_BODY_RAJA;
       //});

     }
     stopTimer();

     INIT_VIEW1D_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  INIT_VIEW1D : Unknown OMP Targetvariant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
