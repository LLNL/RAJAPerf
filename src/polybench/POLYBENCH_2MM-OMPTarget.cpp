
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

#include "POLYBENCH_2MM.hpp"

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
#define NUMTEAMS 256

#define POLYBENCH_2MM_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr tmp; \
  Real_ptr A; \
  Real_ptr B; \
  Real_ptr C; \
  Real_ptr D; \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
  allocAndInitOpenMPDeviceData(tmp, m_tmp, m_ni * m_nj, did, hid); \
  allocAndInitOpenMPDeviceData(A, m_A, m_ni * m_nk, did, hid); \
  allocAndInitOpenMPDeviceData(B, m_B, m_nk * m_nj, did, hid); \
  allocAndInitOpenMPDeviceData(C, m_C, m_nj * m_nl, did, hid); \
  allocAndInitOpenMPDeviceData(D, m_D, m_ni * m_nl, did, hid); 


#define POLYBENCH_2MM_DATA_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_D, D, m_ni * m_nl, hid, did); \
  deallocOpenMPDeviceData(tmp, did); \
  deallocOpenMPDeviceData(A, did); \
  deallocOpenMPDeviceData(B, did); \
  deallocOpenMPDeviceData(C, did); \
  deallocOpenMPDeviceData(D, did);


void POLYBENCH_2MM::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ni = m_ni;
  const Index_type nj = m_nj;
  const Index_type nk = m_nk;
  const Index_type nl = m_nl;

  if ( vid == Base_OpenMPTarget ) {

    POLYBENCH_2MM_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      
      #pragma omp target is_device_ptr(tmp,A,B,C,D) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) collapse(2) 
      for (Index_type i = 0; i < ni; i++ ) {
        for(Index_type j = 0; j < nj; j++) {
          POLYBENCH_2MM_BODY1;
          for(Index_type k = 0; k < nk; k++) {
            POLYBENCH_2MM_BODY2;
          }
        }
      }

      if ( irep == run_reps - 1 ) {
        memset(m_D, 0, m_ni * m_nl * sizeof(Real_type));
      }
      initOpenMPDeviceData(D, m_D, m_ni * m_nl, did, hid); 

      #pragma omp target is_device_ptr(tmp,A,B,C,D) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) collapse(2)
      for(Index_type i = 0; i < ni; i++) {
        for(Index_type l = 0; l < nl; l++) {
          POLYBENCH_2MM_BODY3;
          for(Index_type j = 0; j < nj; j++) {
            POLYBENCH_2MM_BODY4;
          }
        }  
      }

    }
    stopTimer(); 

    POLYBENCH_2MM_DATA_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    POLYBENCH_2MM_DATA_SETUP_OMP_TARGET;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                  RAJA::ArgList<0, 1>,
          RAJA::statement::Lambda<0>,
          RAJA::statement::For<2, RAJA::seq_exec,
            RAJA::statement::Lambda<1>
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                                               RAJA::RangeSegment{0, nj},
                                               RAJA::RangeSegment{0, nk}),
        [=] (Index_type i, Index_type j, Index_type /* k */) {
          POLYBENCH_2MM_BODY1;
        },
        [=] (Index_type i, Index_type j, Index_type k) {
          POLYBENCH_2MM_BODY2;
        }
      );

      if ( irep == run_reps - 1 ) {
        memset(m_D, 0, m_ni * m_nl * sizeof(Real_type));
      }
      initOpenMPDeviceData(D, m_D, m_ni * m_nl, did, hid); 

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                                               RAJA::RangeSegment{0, nl},
                                               RAJA::RangeSegment{0, nj}),
        [=] (Index_type i, Index_type l, Index_type /* j */) {
          POLYBENCH_2MM_BODY3;
        },
        [=] (Index_type i, Index_type l, Index_type j) {
          POLYBENCH_2MM_BODY4;
        }
      );

    }
    stopTimer();

    POLYBENCH_2MM_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  POLYBENCH_2MM : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP

