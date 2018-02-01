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

#include "EOS.hpp"

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

#define EOS_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr x; \
  Real_ptr y; \
  Real_ptr z; \
  Real_ptr u; \
  const Real_type q = m_q; \
  const Real_type r = m_r; \
  const Real_type t = m_t; \
\
  allocAndInitOpenMPDeviceData(x, m_x, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(y, m_y, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(z, m_z, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(u, m_u, m_array_length, did, hid);

#define EOS_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_x, x, m_array_length, hid, did); \
  deallocOpenMPDeviceData(x, did); \
  deallocOpenMPDeviceData(y, did); \
  deallocOpenMPDeviceData(z, did); \
  deallocOpenMPDeviceData(u, did);


void EOS::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_OpenMPTarget ) {

    EOS_DATA_SETUP_OMP_TARGET

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(x, y, z, u) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
      for (Index_type i = ibegin; i < iend; ++i ) {
        EOS_BODY;
      }

    }
    stopTimer();

    EOS_DATA_TEARDOWN_OMP_TARGET

  } else if ( vid == RAJA_OpenMPTarget ) {

    EOS_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        EOS_BODY;
      });  

    }
    stopTimer();

    EOS_DATA_TEARDOWN_OMP_TARGET

  } else { 
     std::cout << "\n  EOS : Unknown OMP Tagretvariant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
