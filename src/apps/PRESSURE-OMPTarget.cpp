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

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

//
// Define thread block size for target execution
//
#define NUMTEAMS 128

#define PRESSURE_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr compression; \
  Real_ptr bvc; \
  Real_ptr p_new; \
  Real_ptr e_old; \
  Real_ptr vnewc; \
  const Real_type cls = m_cls; \
  const Real_type p_cut = m_p_cut; \
  const Real_type pmin = m_pmin; \
  const Real_type eosvmax = m_eosvmax; \
\
  allocAndInitOpenMPDeviceData(compression, m_compression, iend, did, hid); \
  allocAndInitOpenMPDeviceData(bvc, m_bvc, iend, did, hid); \
  allocAndInitOpenMPDeviceData(p_new, m_p_new, iend, did, hid); \
  allocAndInitOpenMPDeviceData(e_old, m_e_old, iend, did, hid); \
  allocAndInitOpenMPDeviceData(vnewc, m_vnewc, iend, did, hid);

#define PRESSURE_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_p_new, p_new, iend, hid, did); \
  deallocOpenMPDeviceData(compression, did); \
  deallocOpenMPDeviceData(bvc, did); \
  deallocOpenMPDeviceData(p_new, did); \
  deallocOpenMPDeviceData(e_old, did); \
  deallocOpenMPDeviceData(vnewc, did);


void PRESSURE::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_OpenMPTarget ) {

    PRESSURE_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(compression, bvc) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
      for (Index_type i = ibegin; i < iend; ++i ) {
        PRESSURE_BODY1;
      }

      #pragma omp target is_device_ptr(bvc, p_new, e_old, vnewc) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
      for (Index_type i = ibegin; i < iend; ++i ) {
        PRESSURE_BODY2;
      }

    }
    stopTimer();

    PRESSURE_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    PRESSURE_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
        RAJA::RangeSegment(ibegin, iend), [=](int i) {
        PRESSURE_BODY1;
      });

      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
        RAJA::RangeSegment(ibegin, iend), [=](int i) {
        PRESSURE_BODY2;
      });

    }
    stopTimer();

    PRESSURE_DATA_TEARDOWN_OMP_TARGET;

  } else {
    std::cout << "\n  PRESSURE : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
