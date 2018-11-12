  
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

#include "POLYBENCH_GEMM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_GEMM_DATA_SETUP_CUDA \
  const Index_type ni = m_ni; \
  const Index_type nj = m_nj; \
  const Index_type nk = m_nk; \
\
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
  Real_ptr A; \
  Real_ptr B; \
  Real_ptr C; \
\
  allocAndInitCudaDeviceData(A, m_A, ni*nk); \
  allocAndInitCudaDeviceData(B, m_B, nk*nj); \
  allocAndInitCudaDeviceData(C, m_C, ni*nj);


#define POLYBENCH_GEMM_TEARDOWN_CUDA \
  getCudaDeviceData(m_C, C, ni*nj); \
  deallocCudaDeviceData(A); \
  deallocCudaDeviceData(B); \
  deallocCudaDeviceData(C);


__global__ void poly_gemm(Real_ptr C, Real_ptr A, Real_ptr B,
                          Real_type alpha, Real_type beta,
                          Index_type nj, Index_type nk) 
{
   Index_type i = blockIdx.x;
   Index_type j = threadIdx.y;

   POLYBENCH_GEMM_BODY1;
   for (Index_type k = 0; k < nk; ++k ) {
     POLYBENCH_GEMM_BODY2;
   }
   POLYBENCH_GEMM_BODY3;
}


void POLYBENCH_GEMM::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  if ( vid == Base_CUDA ) {

    POLYBENCH_GEMM_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nblocks(ni, 1, 1);
      dim3 nthreads_per_block(1, nj, 1);

      poly_gemm<<<nblocks, nthreads_per_block>>>(C, A, B, 
                                                 alpha, beta,
                                                 nj, nk);

    }
    stopTimer();

    POLYBENCH_GEMM_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_GEMM_DATA_SETUP_CUDA;

    POLYBENCH_GEMM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::For<0, RAJA::cuda_block_exec,
            RAJA::statement::For<1, RAJA::cuda_thread_exec,
              RAJA::statement::Lambda<0>,
              RAJA::statement::For<2, RAJA::seq_exec,
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

          RAJA::make_tuple( RAJA::RangeSegment{0, ni},
                            RAJA::RangeSegment{0, nj},
                            RAJA::RangeSegment{0, nk} ),

          RAJA::tuple<double>{0.0},  // variable for dot

          [=] __device__ (Index_type i, Index_type j, Index_type /*k*/, double& dot) {
            POLYBENCH_GEMM_BODY1_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j, Index_type k, double& dot) {
            POLYBENCH_GEMM_BODY2_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j, Index_type /*k*/, double& dot) {
            POLYBENCH_GEMM_BODY3_RAJA;
          }
        );

      }
      stopTimer();

    POLYBENCH_GEMM_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_GEMM : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
  
