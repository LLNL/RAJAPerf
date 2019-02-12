  
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

#include "POLYBENCH_ATAX.hpp"

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

#define POLYBENCH_ATAX_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr tmp; \
  Real_ptr y; \
  Real_ptr x; \
  Real_ptr A; \
\
  allocAndInitOpenMPDeviceData(tmp, m_tmp, N, did, hid); \
  allocAndInitOpenMPDeviceData(y, m_y, N, did, hid); \
  allocAndInitOpenMPDeviceData(x, m_x, N, did, hid); \
  allocAndInitOpenMPDeviceData(A, m_A, N * N, did, hid);


#define POLYBENCH_ATAX_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_y, y, N, hid, did); \
  deallocOpenMPDeviceData(tmp, did); \
  deallocOpenMPDeviceData(y, did); \
  deallocOpenMPDeviceData(x, did); \
  deallocOpenMPDeviceData(A, did);


void POLYBENCH_ATAX::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  if ( vid == Base_OpenMPTarget ) {

    POLYBENCH_ATAX_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(x,y,tmp,A) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1)
      for (Index_type i = 0; i < N; ++i ) {
        POLYBENCH_ATAX_BODY1;
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_ATAX_BODY2;
        }
      }
        
      #pragma omp target is_device_ptr(y,tmp,A) device( did )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1)
      for (Index_type j = 0; j < N; ++j ) {
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_ATAX_BODY3;
        }
      }

    }
    stopTimer();

    POLYBENCH_ATAX_TEARDOWN_OMP_TARGET;

  } else if (vid == RAJA_OpenMPTarget) {

    POLYBENCH_ATAX_DATA_SETUP_OMP_TARGET;

    POLYBENCH_ATAX_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<NUMTEAMS>,
          RAJA::statement::Lambda<0>,
          RAJA::statement::For<1, RAJA::seq_exec,
            RAJA::statement::Lambda<1>
          >
        >
      >;

    using EXEC_POL2 =
      RAJA::KernelPolicy<
        RAJA::statement::For<1, RAJA::omp_target_parallel_for_exec<NUMTEAMS>,
          RAJA::statement::For<0, RAJA::seq_exec,
            RAJA::statement::Lambda<0>
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL1>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                                RAJA::RangeSegment{0, N}),
        [=] (Index_type i, Index_type /* j */) {
          POLYBENCH_ATAX_BODY1_RAJA;
        },
        [=] (Index_type i, Index_type j) {
          POLYBENCH_ATAX_BODY2_RAJA;
        }
      );

      RAJA::kernel<EXEC_POL2>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                                RAJA::RangeSegment{0, N}),
        [=] (Index_type i, Index_type j) {
          POLYBENCH_ATAX_BODY3_RAJA;
        }
      );

    }
    stopTimer();

    POLYBENCH_ATAX_TEARDOWN_OMP_TARGET;

  } else {
      std::cout << "\n  POLYBENCH_ATAX : Unknown OMP Target variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
  
