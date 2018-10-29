  
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

#include "POLYBENCH_JACOBI_1D.hpp"

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

#define POLYBENCH_JACOBI_1D_DATA_SETUP_OMP_TARGET \
  int hid = omp_get_initial_device(); \
  int did = omp_get_default_device(); \
\
  Real_ptr A; \
  Real_ptr B; \
\
  allocAndInitOpenMPDeviceData(A, m_Ainit, m_N, did, hid); \
  allocAndInitOpenMPDeviceData(B, m_Binit, m_N, did, hid);


#define POLYBENCH_JACOBI_1D_TEARDOWN_OMP_TARGET \
  getOpenMPDeviceData(m_A, A, m_N, hid, did); \
  getOpenMPDeviceData(m_B, B, m_N, hid, did); \
  deallocOpenMPDeviceData(A, did); \
  deallocOpenMPDeviceData(B, did);


void POLYBENCH_JACOBI_1D::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;
  const Index_type tsteps = m_tsteps;

  if ( vid == Base_OpenMPTarget ) {

    POLYBENCH_JACOBI_1D_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {
       
        #pragma omp target is_device_ptr(A,B) device( did )
        #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1)
        for (Index_type i = 1; i < N-1; ++i ) {
          POLYBENCH_JACOBI_1D_BODY1;
        }

        #pragma omp target is_device_ptr(A,B) device( did )
        #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1)
        for (Index_type i = 1; i < N-1; ++i ) {
          POLYBENCH_JACOBI_1D_BODY2;
        }
      }

    }
    stopTimer();

    POLYBENCH_JACOBI_1D_TEARDOWN_OMP_TARGET;

  } else if (vid == RAJA_OpenMPTarget ) {

    POLYBENCH_JACOBI_1D_DATA_SETUP_OMP_TARGET;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<NUMTEAMS>,
          RAJA::statement::Lambda<0>
        >,
        RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<NUMTEAMS>,
          RAJA::statement::Lambda<1>
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1}),
          [=] (Index_type i) {
            POLYBENCH_JACOBI_1D_BODY1;
          },
          [=] (Index_type i) {
            POLYBENCH_JACOBI_1D_BODY2;
          }
        );

      }

    }
    stopTimer();

    POLYBENCH_JACOBI_1D_TEARDOWN_OMP_TARGET;

  } else {
      std::cout << "\n  POLYBENCH_JACOBI_1D : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
  
