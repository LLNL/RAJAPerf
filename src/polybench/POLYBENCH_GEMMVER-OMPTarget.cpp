
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

#include "POLYBENCH_GEMMVER.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

//
// Define thread block size for target execution
//
#define NUMTEAMS 128

#define POLYBENCH_GEMMVER_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Index_type n = m_n; \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
  Real_ptr A; \
  Real_ptr u1; \
  Real_ptr v1; \
  Real_ptr u2; \
  Real_ptr v2; \
  Real_ptr w; \
  Real_ptr x; \
  Real_ptr y; \
  Real_ptr z; \
\
  allocAndInitOpenMPDeviceData(A, m_A, m_n * m_n, did, hid); \
  allocAndInitOpenMPDeviceData(u1, m_u1, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(v1, m_v1, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(u2, m_u2, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(v2, m_v2, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(w, m_w, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(x, m_x, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(y, m_y, m_n, did, hid); \
  allocAndInitOpenMPDeviceData(z, m_z, m_n, did, hid); 

#define POLYBENCH_GEMMVER_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_w, w, m_n, hid, did); \
  deallocOpenMPDeviceData(A, did); \
  deallocOpenMPDeviceData(u1, did); \
  deallocOpenMPDeviceData(v1, did); \
  deallocOpenMPDeviceData(u2, did); \
  deallocOpenMPDeviceData(v2, did); \
  deallocOpenMPDeviceData(w, did); \
  deallocOpenMPDeviceData(x, did); \
  deallocOpenMPDeviceData(y, did); \
  deallocOpenMPDeviceData(z, did); 

  

void POLYBENCH_GEMMVER::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type n = m_n;

  if ( vid == Base_OpenMPTarget ) {

    POLYBENCH_GEMMVER_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(A,u1,v1,u2,v2,w,x,y,z) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) collapse(2)
      
      for (Index_type i = 0; i < n; i++ ) {
        for(Index_type j = 0; j < n; j++) {
          POLYBENCH_GEMMVER_BODY1;
        }
      }

      #pragma omp target is_device_ptr(A,u1,v1,u2,v2,w,x,y,z) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) collapse(2) 
      for (Index_type i = 0; i < n; i++ ) { 
        for(Index_type jj = 0; jj < n; jj++) {
           if(jj == 0) {
            for(Index_type j=0; j < n; ++j) { 
              POLYBENCH_GEMMVER_BODY2;
            } 
          }
        }
      }

      #pragma omp target is_device_ptr(A,u1,v1,u2,v2,w,x,y,z) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
      for (Index_type i = 0; i < n; i++ ) {
        POLYBENCH_GEMMVER_BODY3;
      }

      #pragma omp target is_device_ptr(A,u1,v1,u2,v2,w,x,y,z) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) collapse(2)
      for (Index_type i = 0; i < n; i++ ) {
        for(Index_type jj = 0; jj < n; jj++) {
           if(jj == 0) {
            for(Index_type j=0; j < n; ++j) { 
              POLYBENCH_GEMMVER_BODY4;
            } 
          }   
        }
      }

    } // end run_reps
    stopTimer(); 
    POLYBENCH_GEMMVER_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    POLYBENCH_GEMMVER_DATA_SETUP_OMP_TARGET;

    startTimer();

    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::policy::omp::omp_target_parallel_for_exec<NUMTEAMS>>(
          RAJA::RangeSegment(0,n * n), [=](Index_type ii) {
        Index_type i,j;
        i = ii/n; j = ii % n;
        POLYBENCH_GEMMVER_BODY1; 
      });

      RAJA::forall<RAJA::policy::omp::omp_target_parallel_for_exec<NUMTEAMS>>(
          RAJA::RangeSegment(0,n * n), [=](Index_type ii) {
          Index_type i,jj;
          i = ii/n; jj = ii % n;
          if(jj == 0) {
            for(Index_type j=0; j < n; j++) {
              POLYBENCH_GEMMVER_BODY2;
            } 
          }
      });

      RAJA::forall<RAJA::policy::omp::omp_target_parallel_for_exec<NUMTEAMS>>(
          RAJA::RangeSegment(0,n), [=](Index_type i) {
        POLYBENCH_GEMMVER_BODY3; 
      });

      RAJA::forall<RAJA::policy::omp::omp_target_parallel_for_exec<NUMTEAMS>>(
          RAJA::RangeSegment(0,n * n), [=](Index_type ii) {
          Index_type i,jj;
          i = ii/n; jj = ii % n;
          if(jj == 0) {
            for(Index_type j=0; j < n; j++) { 
              POLYBENCH_GEMMVER_BODY4;
            } 
          }   
      });

    } // for run_reps
    stopTimer();
    POLYBENCH_GEMMVER_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  POLYBENCH_GEMMVER : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

