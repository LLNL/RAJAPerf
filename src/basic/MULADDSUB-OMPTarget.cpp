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

#include "MULADDSUB.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

//
// Define thread block size for target execution
//
#define NUMTEAMS 128

#define MULADDSUB_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr out1; \
  Real_ptr out2; \
  Real_ptr out3; \
  Real_ptr in1; \
  Real_ptr in2; \
\
  allocAndInitOpenMPDeviceData(out1, m_out1, iend, did, hid); \
  allocAndInitOpenMPDeviceData(out2, m_out2, iend, did, hid); \
  allocAndInitOpenMPDeviceData(out3, m_out3, iend, did, hid); \
  allocAndInitOpenMPDeviceData(in1, m_in1, iend, did, hid); \
  allocAndInitOpenMPDeviceData(in2, m_in2, iend, did, hid);

#define MULADDSUB_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_out1, out1, iend, hid, did); \
  getOpenMPDeviceData(m_out2, out2, iend, hid, did); \
  getOpenMPDeviceData(m_out3, out3, iend, hid, did); \
  deallocOpenMPDeviceData(out1, did); \
  deallocOpenMPDeviceData(out2, did); \
  deallocOpenMPDeviceData(out3, did); \
  deallocOpenMPDeviceData(in1, did); \
  deallocOpenMPDeviceData(in2, did);


void MULADDSUB::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_OpenMPTarget ) {

    MULADDSUB_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(out1, out2, out3, in1, in2) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        MULADDSUB_BODY;
      }

    }
    stopTimer();

    MULADDSUB_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    MULADDSUB_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        MULADDSUB_BODY;
      });

    }
    stopTimer();

    MULADDSUB_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  MULADDSUB : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
