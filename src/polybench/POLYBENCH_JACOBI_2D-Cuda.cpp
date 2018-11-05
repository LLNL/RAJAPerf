  
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

#include "POLYBENCH_JACOBI_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_JACOBI_2D_DATA_SETUP_CUDA \
  Real_ptr A; \
  Real_ptr B; \
\
  allocAndInitCudaDeviceData(A, m_Ainit, m_N*m_N); \
  allocAndInitCudaDeviceData(B, m_Binit, m_N*m_N);


#define POLYBENCH_JACOBI_2D_TEARDOWN_CUDA \
  getCudaDeviceData(m_A, A, m_N*m_N); \
  getCudaDeviceData(m_B, B, m_N*m_N); \
  deallocCudaDeviceData(A); \
  deallocCudaDeviceData(B);


__global__ void poly_jacobi_2D_1(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = blockIdx.x;
   Index_type j = threadIdx.y;

   if ( i > 0 && j > 0 && i < N-1 && j < N-1 ) {
     POLYBENCH_JACOBI_2D_BODY1;
   }
}

__global__ void poly_jacobi_2D_2(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = blockIdx.x;
   Index_type j = threadIdx.y;

   if ( i > 0 && j > 0 && i < N-1 && j < N-1 ) {
     POLYBENCH_JACOBI_2D_BODY2;
   }
}


void POLYBENCH_JACOBI_2D::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;
  const Index_type tsteps = m_tsteps;

  if ( vid == Base_CUDA ) {

    POLYBENCH_JACOBI_2D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        dim3 nblocks(N, 1, 1);
        dim3 nthreads_per_block(1, N, 1);

        poly_jacobi_2D_1<<<nblocks, nthreads_per_block>>>(A, B, N);

        poly_jacobi_2D_2<<<nblocks, nthreads_per_block>>>(A, B, N);

      }

    }
    stopTimer();

    POLYBENCH_JACOBI_2D_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_JACOBI_2D_DATA_SETUP_CUDA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::For<0, RAJA::cuda_block_exec,
            RAJA::statement::For<1, RAJA::cuda_thread_exec,
              RAJA::statement::Lambda<0>
            >
          >
        >,
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::For<0, RAJA::cuda_block_exec,
            RAJA::statement::For<1, RAJA::cuda_thread_exec,
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
            POLYBENCH_JACOBI_2D_BODY1;
          },
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY2;
          }
        );

      }

    }
    stopTimer();

    POLYBENCH_JACOBI_2D_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_JACOBI_2D : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
  
