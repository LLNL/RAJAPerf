  
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
// For details about use and distribution, please read RAJAPerf/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_MVT.hpp"

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

#define POLYBENCH_MVT_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr x1; \
  Real_ptr x2; \
  Real_ptr y1; \
  Real_ptr y2; \
  Real_ptr A; \
\
  allocAndInitOpenMPDeviceData(x1, m_x1, N, did, hid); \
  allocAndInitOpenMPDeviceData(x2, m_x2, N, did, hid); \
  allocAndInitOpenMPDeviceData(y1, m_y1, N, did, hid); \
  allocAndInitOpenMPDeviceData(y2, m_y2, N, did, hid); \
  allocAndInitOpenMPDeviceData(A, m_A, N * N, did, hid);


#define POLYBENCH_MVT_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_x1, x1, N, hid, did); \
  getOpenMPDeviceData(m_x2, x2, N, hid, did); \
  deallocOpenMPDeviceData(x1, did); \
  deallocOpenMPDeviceData(x2, did); \
  deallocOpenMPDeviceData(y1, did); \
  deallocOpenMPDeviceData(y2, did); \
  deallocOpenMPDeviceData(A, did);


void POLYBENCH_MVT::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  if ( vid == Base_OpenMPTarget ) {

    POLYBENCH_MVT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(x1,A,y1) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1)
      for (Index_type i = 0; i < N; ++i ) {
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_MVT_BODY1;
        }
      }

      #pragma omp target is_device_ptr(x2,A,y2) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1)
      for (Index_type i = 0; i < N; ++i ) {
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_MVT_BODY2;
        }
      }

    }
    stopTimer();

    POLYBENCH_MVT_TEARDOWN_OMP_TARGET;

  } else if (vid == RAJA_OpenMPTarget) {

    POLYBENCH_MVT_DATA_SETUP_OMP_TARGET;

    POLYBENCH_MVT_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<NUMTEAMS>,
          RAJA::statement::For<1, RAJA::seq_exec,
            RAJA::statement::Lambda<0>
          >
        >,
        RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<NUMTEAMS>,
          RAJA::statement::For<1, RAJA::seq_exec,
            RAJA::statement::Lambda<1>
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                               RAJA::RangeSegment{0, N}),
        [=] (Index_type i, Index_type j) {
          POLYBENCH_MVT_BODY1_RAJA;
        },
        [=] (Index_type i, Index_type j) {
          POLYBENCH_MVT_BODY2_RAJA;
        }
      );

    }
    stopTimer();

    POLYBENCH_MVT_TEARDOWN_OMP_TARGET;

  } else {
      std::cout << "\n  POLYBENCH_MVT : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
  
