  
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

#include "POLYBENCH_GEMVER.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

//
// Define thread block size for CUDA execution
//
const size_t block_size = 256;

#define POLYBENCH_GEMVER_DATA_SETUP_CUDA \
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
  allocAndInitCudaDeviceData(A, m_A, m_n * m_n); \
  allocAndInitCudaDeviceData(u1, m_u1, m_n); \
  allocAndInitCudaDeviceData(v1, m_v1, m_n); \
  allocAndInitCudaDeviceData(u2, m_u2, m_n); \
  allocAndInitCudaDeviceData(v2, m_v2, m_n); \
  allocAndInitCudaDeviceData(w, m_w, m_n); \
  allocAndInitCudaDeviceData(x, m_x, m_n); \
  allocAndInitCudaDeviceData(y, m_y, m_n); \
  allocAndInitCudaDeviceData(z, m_z, m_n); 


#define POLYBENCH_GEMVER_TEARDOWN_CUDA \
  getCudaDeviceData(m_w, w, m_n); \
  deallocCudaDeviceData(A); \
  deallocCudaDeviceData(u1); \
  deallocCudaDeviceData(v1); \
  deallocCudaDeviceData(u2); \
  deallocCudaDeviceData(v2); \
  deallocCudaDeviceData(w); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(z); 

__global__ void poly_gemmver_1(Real_ptr A, 
                               Real_ptr u1, Real_ptr v1, 
                               Real_ptr u2, Real_ptr v2,
                               Index_type n)
{
  Index_type i = blockIdx.x;
  Index_type j = threadIdx.y; 

  POLYBENCH_GEMVER_BODY1;
}

__global__ void poly_gemmver_2(Real_ptr A, 
                               Real_ptr x, Real_ptr y,
                               Real_type beta, 
                               Index_type n)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { 
    for (Index_type j = 0; j < n; ++j) {
      POLYBENCH_GEMVER_BODY2;
    }
  }
}

__global__ void poly_gemmver_3(Real_ptr x, Real_ptr z,
                               Index_type n)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    POLYBENCH_GEMVER_BODY3;
  }
}

__global__ void poly_gemmver_4(Real_ptr A,
                               Real_ptr x, Real_ptr w,
                               Real_type alpha,
                               Index_type n)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { 
    for (Index_type j = 0; j < n; ++j) {
      POLYBENCH_GEMVER_BODY4;
    }
  }
}


void POLYBENCH_GEMVER::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  
  if ( vid == Base_CUDA ) {

    POLYBENCH_GEMVER_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nblocks(n, 1, 1);
      dim3 nthreads_per_block(1, n, 1);
      poly_gemmver_1<<<nblocks, nthreads_per_block>>>(A, u1, v1, u2, v2, 
                                                      n);

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_n, block_size); 

      poly_gemmver_2<<<grid_size, block_size>>>(A, x, y,
                                                beta,
                                                n);

      poly_gemmver_3<<<grid_size, block_size>>>(x, z,
                                                n); 

      poly_gemmver_4<<<grid_size, block_size>>>(A, x, w,
                                                alpha,
                                                n);
    }
    stopTimer();

    POLYBENCH_GEMVER_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_GEMVER_DATA_SETUP_CUDA;

    POLYBENCH_GEMVER_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::For<0, RAJA::cuda_block_x_loop,
            RAJA::statement::For<1, RAJA::cuda_thread_y_loop,
              RAJA::statement::Lambda<0>,
            >
          >
        >
      >;

    using EXEC_POL24 = 
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<0>,
            >
          >
        >
      >;
 
    using EXEC_POL3 = RAJA::cuda_exec<block_size, true /*async*/>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL1>( RAJA::make_tuple(RAJA::RangeSegment{0, n},
                                                RAJA::RangeSegment{0, n}),
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_GEMVER_BODY1;
        }
      );

      RAJA::kernel<EXEC_POL24>( RAJA::make_tuple(RAJA::RangeSegment{0, n},
                                                 RAJA::RangeSegment{0, n}),
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_GEMVER_BODY2;
        }
      );

      RAJA::forall<EXEC_POL3> (
        RAJA::RangeSegment{0, n}, [=] __device__ (Index_type i) {
        POLYBENCH_GEMVER_BODY3;
      });

      RAJA::kernel<EXEC_POL24>( RAJA::make_tuple(RAJA::RangeSegment{0, n},
                                                 RAJA::RangeSegment{0, n}),
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_GEMVER_BODY4;
        }
      );
      
    }
    stopTimer();

    POLYBENCH_GEMVER_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_GEMVER : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
  
