  
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

#define POLYBENCH_MVT_DATA_SETUP_CUDA \
  Real_ptr x1; \
  Real_ptr x2; \
  Real_ptr y1; \
  Real_ptr y2; \
  Real_ptr A; \
\
  allocAndInitCudaDeviceData(x1, m_x1, N); \
  allocAndInitCudaDeviceData(x2, m_x2, N); \
  allocAndInitCudaDeviceData(y1, m_y1, N); \
  allocAndInitCudaDeviceData(y2, m_y2, N); \
  allocAndInitCudaDeviceData(A, m_A, N * N);


#define POLYBENCH_MVT_TEARDOWN_CUDA \
  getCudaDeviceData(m_x1, x1, N); \
  getCudaDeviceData(m_x2, x2, N); \
  deallocCudaDeviceData(x1); \
  deallocCudaDeviceData(x2); \
  deallocCudaDeviceData(y1); \
  deallocCudaDeviceData(y2); \
  deallocCudaDeviceData(A);


__global__ void poly_mvt_1(Real_ptr A, Real_ptr x1, Real_ptr y1,
                           Index_type N) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < N) {
     double dot;
     POLYBENCH_MVT_BODY1a;
     for (Index_type j = 0; j < N; ++j ) {
       POLYBENCH_MVT_BODY1b;
     }
     POLYBENCH_MVT_BODY1c;
   }
}

__global__ void poly_mvt_2(Real_ptr A, Real_ptr x2, Real_ptr y2,
                           Index_type N) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < N) {
     double dot;
     POLYBENCH_MVT_BODY2a;
     for (Index_type j = 0; j < N; ++j ) {
       POLYBENCH_MVT_BODY2b;
     }
     POLYBENCH_MVT_BODY2c;
   }
}


void POLYBENCH_MVT::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  if ( vid == Base_CUDA ) {

    POLYBENCH_MVT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);

      poly_mvt_1<<<grid_size, block_size>>>(A, x1, y1, N);

      poly_mvt_2<<<grid_size, block_size>>>(A, x2, y2, N);

    }
    stopTimer();

    POLYBENCH_MVT_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_MVT_DATA_SETUP_CUDA;

    POLYBENCH_MVT_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::Tile<0, RAJA::statement::tile_fixed<block_size>, RAJA::cuda_block_x_loop,
            RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
              RAJA::statement::Lambda<0>,
                RAJA::statement::For<1, RAJA::seq_exec,
                  RAJA::statement::Lambda<1>
                >,
                RAJA::statement::Lambda<2>
            >
          >
        >,
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::Tile<0, RAJA::statement::tile_fixed<block_size>, RAJA::cuda_block_x_loop,
            RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
              RAJA::statement::Lambda<3>,
                RAJA::statement::For<1, RAJA::seq_exec,
                  RAJA::statement::Lambda<4>
                >,
                RAJA::statement::Lambda<5>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel_param<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                                     RAJA::RangeSegment{0, N}),
                                    RAJA::make_tuple(0.0),
                                    [=] __device__ (Index_type i, Index_type j, double &dot) {
                                      POLYBENCH_MVT_BODY1a_RAJA;
                                    },

                                    [=] __device__ (Index_type i, Index_type j, double &dot) {
                                      POLYBENCH_MVT_BODY1b_RAJA;
                                    },

                                    [=] __device__ (Index_type i, Index_type j, double &dot) {
                                      POLYBENCH_MVT_BODY1c_RAJA;
                                    },

                                    [=] __device__ (Index_type i, Index_type j, double &dot) {
                                      POLYBENCH_MVT_BODY2a_RAJA;
                                    },

                                    [=] __device__ (Index_type i, Index_type j, double &dot) {
                                      POLYBENCH_MVT_BODY2b_RAJA;
                                    },

                                    [=] __device__ (Index_type i, Index_type j, double &dot) {
                                      POLYBENCH_MVT_BODY2c_RAJA;
                                    }                                    
      );

    }
    stopTimer();

    POLYBENCH_MVT_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_MVT : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
  
