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

#include "DEL_DOT_VEC_2D.hpp"

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

#define DEL_DOT_VEC_2D_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_ptr xdot = m_xdot; \
  Real_ptr ydot = m_ydot; \
  Real_ptr div = m_div; \
  Index_ptr real_zones; \
\
  const Real_type ptiny = m_ptiny; \
  const Real_type half = m_half; \
\
  Real_ptr x1,x2,x3,x4 ; \
  Real_ptr y1,y2,y3,y4 ; \
  Real_ptr fx1,fx2,fx3,fx4 ; \
  Real_ptr fy1,fy2,fy3,fy4 ; \
\
  allocAndInitOpenMPDeviceData(x, m_x, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(y, m_y, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(xdot, m_xdot, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(ydot, m_ydot, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(div, m_div, m_array_length, did, hid); \
  allocAndInitOpenMPDeviceData(real_zones, m_domain->real_zones, iend, did, hid);

#define DEL_DOT_VEC_2D_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_div, div, m_array_length, hid, did); \
  deallocOpenMPDeviceData(x, did); \
  deallocOpenMPDeviceData(y, did); \
  deallocOpenMPDeviceData(xdot, did); \
  deallocOpenMPDeviceData(ydot, did); \
  deallocOpenMPDeviceData(div, did); \
  deallocOpenMPDeviceData(real_zones, did);


void DEL_DOT_VEC_2D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  if ( vid == Base_OpenMPTarget ) {

    DEL_DOT_VEC_2D_DATA_SETUP_OMP_TARGET;
     
    NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
    NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
    NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
    NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(x1,x2,x3,x4, y1,y2,y3,y4, \
                                       fx1,fx2,fx3,fx4, fy1,fy2,fy3,fy4, \
                                       div, real_zones) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
      for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
        DEL_DOT_VEC_2D_BODY_INDEX;
        DEL_DOT_VEC_2D_BODY;
      }

    }
    stopTimer();

    DEL_DOT_VEC_2D_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    DEL_DOT_VEC_2D_DATA_SETUP_OMP_TARGET;
     
    NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
    NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
    NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
    NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

#if 0
// we need to fix the fact that list segment data is in UM iff CUDA is 
// enabled...
    RAJA::ListSegment zones(m_domain->real_zones, m_domain->n_real_zones);  
#endif

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::policy::omp::omp_target_parallel_for_exec<NUMTEAMS>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type ii) {
        DEL_DOT_VEC_2D_BODY_INDEX;
        DEL_DOT_VEC_2D_BODY;
      });

    }
    stopTimer();

    DEL_DOT_VEC_2D_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  DEL_DOT_VEC_2D : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
