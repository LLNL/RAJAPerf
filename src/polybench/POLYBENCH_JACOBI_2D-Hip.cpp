//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//  

#include "POLYBENCH_JACOBI_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_JACOBI_2D_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(A, m_Ainit, m_N*m_N); \
  allocAndInitHipDeviceData(B, m_Binit, m_N*m_N);


#define POLYBENCH_JACOBI_2D_TEARDOWN_HIP \
  getHipDeviceData(m_A, A, m_N*m_N); \
  getHipDeviceData(m_B, B, m_N*m_N); \
  deallocHipDeviceData(A); \
  deallocHipDeviceData(B);


__global__ void poly_jacobi_2D_1(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = blockIdx.y;
   Index_type j = threadIdx.x;

   if ( i > 0 && j > 0 && i < N-1 && j < N-1 ) {
     POLYBENCH_JACOBI_2D_BODY1;
   }
}

__global__ void poly_jacobi_2D_2(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = blockIdx.y;
   Index_type j = threadIdx.x;

   if ( i > 0 && j > 0 && i < N-1 && j < N-1 ) {
     POLYBENCH_JACOBI_2D_BODY2;
   }
}


void POLYBENCH_JACOBI_2D::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_JACOBI_2D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        dim3 nblocks(1, N, 1);
        dim3 nthreads_per_block(N, 1, 1);

        hipLaunchKernelGGL((poly_jacobi_2D_1),dim3(nblocks), dim3(nthreads_per_block),0,0,A, B, N);

        hipLaunchKernelGGL((poly_jacobi_2D_2),dim3(nblocks), dim3(nthreads_per_block),0,0,A, B, N);

      }

    }
    stopTimer();

    POLYBENCH_JACOBI_2D_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_JACOBI_2D_DATA_SETUP_HIP;

    POLYBENCH_JACOBI_2D_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelAsync<
          RAJA::statement::For<0, RAJA::hip_block_y_loop,
            RAJA::statement::For<1, RAJA::hip_thread_x_loop,
              RAJA::statement::Lambda<0>
            >
          >
        >,
        RAJA::statement::HipKernelAsync<
          RAJA::statement::For<0, RAJA::hip_block_y_loop,
            RAJA::statement::For<1, RAJA::hip_thread_x_loop,
              RAJA::statement::Lambda<1>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                 RAJA::RangeSegment{1, N-1}),
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY1_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY2_RAJA;
          }
        );

      }

    }
    stopTimer();

    POLYBENCH_JACOBI_2D_TEARDOWN_HIP;

  } else {
      std::cout << "\n  POLYBENCH_JACOBI_2D : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
  
