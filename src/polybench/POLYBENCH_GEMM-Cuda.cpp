//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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

//
// Define thread block size for CUDA execution
//
constexpr size_t i_block_sz = 8;
constexpr size_t j_block_sz = 32;

#define POLY_GEMM_THREADS_PER_BLOCK_CUDA \
  dim3 nthreads_per_block(j_block_sz, i_block_sz, 1);

#define POLY_GEMM_NBLOCKS_CUDA \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, j_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, i_block_sz)), \
               static_cast<size_t>(1));


#define POLYBENCH_GEMM_DATA_SETUP_CUDA \
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
                          Index_type ni, Index_type nj, Index_type nk)
{
  Index_type i = blockIdx.y * blockDim.y + threadIdx.y;
  Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i < ni && j < nj ) {
    POLYBENCH_GEMM_BODY1;
    POLYBENCH_GEMM_BODY2;
    for (Index_type k = 0; k < nk; ++k ) {
      POLYBENCH_GEMM_BODY3;
    }
    POLYBENCH_GEMM_BODY4;
  }
}

template< typename Lambda >
__global__ void poly_gemm_lam(Index_type ni, Index_type nj,
                              Lambda body)
{
  Index_type i = blockIdx.y * blockDim.y + threadIdx.y;
  Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i < ni && j < nj ) {
    body(i, j);
  }
}


void POLYBENCH_GEMM::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_GEMM_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    POLYBENCH_GEMM_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      POLY_GEMM_THREADS_PER_BLOCK_CUDA;
      POLY_GEMM_NBLOCKS_CUDA; 

      poly_gemm<<<nblocks, nthreads_per_block>>>(C, A, B,
                                                 alpha, beta,
                                                 ni, nj, nk);
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    POLYBENCH_GEMM_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    POLYBENCH_GEMM_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      POLY_GEMM_THREADS_PER_BLOCK_CUDA;
      POLY_GEMM_NBLOCKS_CUDA;

      poly_gemm_lam<<<nblocks, nthreads_per_block>>>(ni, nj,
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_GEMM_BODY1;
          POLYBENCH_GEMM_BODY2;
          for (Index_type k = 0; k < nk; ++k ) {
            POLYBENCH_GEMM_BODY3;
          }
          POLYBENCH_GEMM_BODY4;
        }
      );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    POLYBENCH_GEMM_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_GEMM_DATA_SETUP_CUDA;

    POLYBENCH_GEMM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixedAsync<i_block_sz * j_block_sz,
          RAJA::statement::Tile<0, RAJA::tile_fixed<i_block_sz>,
                                   RAJA::cuda_block_y_direct,
            RAJA::statement::Tile<1, RAJA::tile_fixed<j_block_sz>,
                                     RAJA::cuda_block_x_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_y_direct,   // i
                RAJA::statement::For<1, RAJA::cuda_thread_x_direct, // j
                  RAJA::statement::Lambda<0, RAJA::Params<0>>,
                  RAJA::statement::Lambda<1, RAJA::Segs<0,1>>,
                  RAJA::statement::For<2, RAJA::seq_exec,           // k
                    RAJA::statement::Lambda<2, RAJA::Segs<0,1,2>, RAJA::Params<0>>
                  >,
                  RAJA::statement::Lambda<3, RAJA::Segs<0,1>, RAJA::Params<0>>
                >
              >
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
          RAJA::tuple<Real_type>{0.0},   // variable for dot

          [=] __device__ (Real_type& dot) {
            POLYBENCH_GEMM_BODY1_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_GEMM_BODY2_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j, Index_type k,
                          Real_type& dot) {
            POLYBENCH_GEMM_BODY3_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j,
                          Real_type& dot) {
            POLYBENCH_GEMM_BODY4_RAJA;
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

