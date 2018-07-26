
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

#include "POLYBENCH_ADI.hpp"

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
#define POLYBENCH_ADI_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr U = m_U; \
  Real_ptr V = m_V; \
  Real_ptr P = m_P; \
  Real_ptr Q = m_Q; \
\
  allocAndInitOpenMPDeviceData(U, m_U, m_n * m_n, did, hid); \
  allocAndInitOpenMPDeviceData(V, m_V, m_n * m_n, did, hid); \
  allocAndInitOpenMPDeviceData(P, m_P, m_n * m_n, did, hid); \
  allocAndInitOpenMPDeviceData(Q, m_Q, m_n * m_n, did, hid); 

#define POLYBENCH_ADI_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_U, U, m_n * m_n, hid, did); \
  deallocOpenMPDeviceData(U, did); \
  deallocOpenMPDeviceData(V, did); \
  deallocOpenMPDeviceData(P, did); \
  deallocOpenMPDeviceData(Q, did); 

void POLYBENCH_ADI::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type n = m_n;
  const Index_type tsteps = m_tsteps;
  Index_type i,j,t; 

  if ( vid == Base_OpenMPTarget ) {

    POLYBENCH_ADI_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      POLYBENCH_ADI_BODY1;
      for (t = 1; t <= tsteps; t++ ) { 
        #pragma omp target is_device_ptr(U,V,P,Q) device( did )
        #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1)
        for(i = 1; i < n-1; i++) {
          POLYBENCH_ADI_BODY2;
          for(j = 1; j < n-1; j++) {
            POLYBENCH_ADI_BODY3;
          }  
          POLYBENCH_ADI_BODY4;
          for(j = 1; j < n-1; j++) {
            POLYBENCH_ADI_BODY5;
          }  
        }

        #pragma omp target is_device_ptr(U,V,P,Q) device( did )
        #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1)
        for(Index_type i = 1; i < n-1; i++) {
          POLYBENCH_ADI_BODY6;
          for(j = 1; j < n-1; j++) {
            POLYBENCH_ADI_BODY7;
          }
          POLYBENCH_ADI_BODY8;
          for(j = 1; j < n-1; j++) {
            POLYBENCH_ADI_BODY9;
          }  
        }
      } // tsteps
    } // run_reps  
    stopTimer(); 
    POLYBENCH_ADI_DATA_TEARDOWN_OMP_TARGET;  
  } else if ( vid == RAJA_OpenMPTarget ) {

    POLYBENCH_ADI_DATA_SETUP_OMP_TARGET;

    startTimer();

    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      POLYBENCH_ADI_BODY1;
      RAJA::forall<RAJA::seq_exec> (
        RAJA::RangeSegment(1, tsteps+1), [=](Index_type t) { 
          RAJA::forall<RAJA::policy::omp::omp_target_parallel_for_exec<NUMTEAMS>>(
            RAJA::RangeSegment(1,n - 1), [=](Index_type i) {
            Index_type j;
            POLYBENCH_ADI_BODY2;
            for(j = 1; j < n-1; j++) {
              POLYBENCH_ADI_BODY3;
            }  
            POLYBENCH_ADI_BODY4;
            for(j = 1; j < n-1; j++) {
              POLYBENCH_ADI_BODY5;
            }  
          });
          RAJA::forall<RAJA::policy::omp::omp_target_parallel_for_exec<NUMTEAMS>>(
            RAJA::RangeSegment(1,n - 1), [=](Index_type i) {
            Index_type j;
            POLYBENCH_ADI_BODY6;
            for(j = 1; j < n-1; j++) {
              POLYBENCH_ADI_BODY7;
            }  
            POLYBENCH_ADI_BODY8;
            for(j = 1; j < n-1; j++) {
              POLYBENCH_ADI_BODY9;
            }  
          });
        }); // tsteps
    } // run_reps

    stopTimer();
    POLYBENCH_ADI_DATA_TEARDOWN_OMP_TARGET;
  } else {
     std::cout << "\n  POLYBENCH_ADI : Unknown OMP Target variant id = " << vid << std::endl;
  }
}    

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

