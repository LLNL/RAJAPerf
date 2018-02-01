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

#include "IF_QUAD.hpp"

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

#define IF_QUAD_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr a; \
  Real_ptr b; \
  Real_ptr c; \
  Real_ptr x1; \
  Real_ptr x2; \
\
  allocAndInitOpenMPDeviceData(a, m_a, iend, did, hid); \
  allocAndInitOpenMPDeviceData(b, m_b, iend, did, hid); \
  allocAndInitOpenMPDeviceData(c, m_c, iend, did, hid); \
  allocAndInitOpenMPDeviceData(x1, m_x1, iend, did, hid); \
  allocAndInitOpenMPDeviceData(x2, m_x2, iend, did, hid);

#define IF_QUAD_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_x1, x1, iend, hid, did); \
  getOpenMPDeviceData(m_x2, x2, iend, hid, did); \
  deallocOpenMPDeviceData(a, did); \
  deallocOpenMPDeviceData(b, did); \
  deallocOpenMPDeviceData(c, did); \
  deallocOpenMPDeviceData(x1, did); \
  deallocOpenMPDeviceData(x2, did);

void IF_QUAD::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_OpenMPTarget ) {

    IF_QUAD_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(a, b, c, x1, x2) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
      for (Index_type i = ibegin; i < iend; ++i ) {
        IF_QUAD_BODY;
      }

    }
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    IF_QUAD_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        IF_QUAD_BODY;
      });

    }
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  IF_QUAD : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
