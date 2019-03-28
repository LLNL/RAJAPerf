  
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
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

#include "POLYBENCH_GESUMMV.hpp"

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

#define POLYBENCH_GESUMMV_DATA_SETUP_CUDA \
  const Index_type N = m_N; \
\
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
  Real_ptr x; \
  Real_ptr y; \
  Real_ptr A; \
  Real_ptr B; \
\
  allocAndInitCudaDeviceData(x, m_x, N); \
  allocAndInitCudaDeviceData(y, m_y, N); \
  allocAndInitCudaDeviceData(A, m_A, N*N); \
  allocAndInitCudaDeviceData(B, m_B, N*N);


#define POLYBENCH_GESUMMV_TEARDOWN_CUDA \
  getCudaDeviceData(m_y, y, N); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(A); \
  deallocCudaDeviceData(B);


__global__ void poly_gesummv(Real_ptr x, Real_ptr y,
                             Real_ptr A, Real_ptr B,
                             Real_type alpha, Real_type beta,
                             Index_type N) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < N) {
     POLYBENCH_GESUMMV_BODY1;
     for (Index_type j = 0; j < N; ++j ) {
       POLYBENCH_GESUMMV_BODY2;
     }
     POLYBENCH_GESUMMV_BODY3;
   }
}


void POLYBENCH_GESUMMV::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  if ( vid == Base_CUDA ) {

    POLYBENCH_GESUMMV_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);

      poly_gesummv<<<grid_size, block_size>>>(x, y, 
                                              A, B, 
                                              alpha, beta,
                                              N);

    }
    stopTimer();

    POLYBENCH_GESUMMV_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_GESUMMV_DATA_SETUP_CUDA;

    POLYBENCH_GESUMMV_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::Tile<0, RAJA::statement::tile_fixed<block_size>, RAJA::cuda_block_x_loop,
            RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
              RAJA::statement::Lambda<0>,
              RAJA::statement::For<1, RAJA::seq_exec,
                RAJA::statement::Lambda<1>
              >,
              RAJA::statement::Lambda<2>
            >
          >
        >
      >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple( RAJA::RangeSegment{0, N},
                            RAJA::RangeSegment{0, N} ),
          RAJA::make_tuple(static_cast<Real_type>(0.0),
                           static_cast<Real_type>(0.0)),

          [=] __device__ (Index_type /*i*/, Index_type /*j*/, Real_type& tmpdot,
                                                              Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY1_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j, Real_type& tmpdot,
                                                      Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY2_RAJA;
          },
          [=] __device__ (Index_type i, Index_type /*j*/, Real_type& tmpdot,
                                                          Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY3_RAJA;
          }
        );

      }
      stopTimer();

    POLYBENCH_GESUMMV_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_GESUMMV : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
  
