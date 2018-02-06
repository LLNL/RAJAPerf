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

#include "VOL3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

//
// Define thread block size for target execution
//
#define NUMTEAMS 128

#define VOL3D_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr x; \
  Real_ptr y; \
  Real_ptr z; \
  Real_ptr vol; \
\
  const Real_type vnormq = m_vnormq; \
\
  Real_ptr x0,x1,x2,x3,x4,x5,x6,x7 ; \
  Real_ptr y0,y1,y2,y3,y4,y5,y6,y7 ; \
  Real_ptr z0,z1,z2,z3,z4,z5,z6,z7 ; \
\
  allocAndInitOpenMPDeviceData(x, m_x, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(y, m_y, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(z, m_z, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(vol, m_vol, m_array_length, did, hid);

#define VOL3D_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_vol, vol, m_array_length, hid, did); \
  deallocOpenMPDeviceData(x, did); \
  deallocOpenMPDeviceData(y, did); \
  deallocOpenMPDeviceData(z, did); \
  deallocOpenMPDeviceData(vol, did);


void VOL3D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = m_domain->fpz;
  const Index_type iend = m_domain->lpz+1;

  if ( vid == Base_OpenMPTarget ) {

    VOL3D_DATA_SETUP_OMP_TARGET;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(x0,x1,x2,x3,x4,x5,x6,x7, \
                                       y0,y1,y2,y3,y4,y5,y6,y7, \
                                       z0,z1,z2,z3,z4,z5,z6,z7, \
                                       vol) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
      for (Index_type i = ibegin ; i < iend ; ++i ) {
        VOL3D_BODY;
      }

    }
    stopTimer();

    VOL3D_DATA_TEARDOWN_OMP_TARGET;     

  } else if ( vid == RAJA_OpenMPTarget ) {

    VOL3D_DATA_SETUP_OMP_TARGET;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {

        VOL3D_BODY;
      });

    }
    stopTimer();

    VOL3D_DATA_TEARDOWN_OMP_TARGET;     

  } else {
    std::cout << "\n  VOL3D : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
