//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DIFF_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

//
// Define thread block size for target execution
//
#define NUMTEAMS 128

#define DIFF_PREDICT_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr px; \
  Real_ptr cx; \
  const Index_type offset = m_offset; \
\
  allocAndInitOpenMPDeviceData(px, m_px, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(cx, m_cx, m_array_length, did, hid);

#define DIFF_PREDICT_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_px, px, m_array_length, hid, did); \
  deallocOpenMPDeviceData(px, did); \
  deallocOpenMPDeviceData(cx, did);


void DIFF_PREDICT::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_OpenMPTarget ) {

    DIFF_PREDICT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(px, cx) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
      for (Index_type i = ibegin; i < iend; ++i ) {
        DIFF_PREDICT_BODY;
      }

    }
    stopTimer();

    DIFF_PREDICT_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    DIFF_PREDICT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        DIFF_PREDICT_BODY;
      });

    }
    stopTimer();

    DIFF_PREDICT_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  DIFF_PREDICT : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
