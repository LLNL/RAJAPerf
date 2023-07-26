//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
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
// Define thread block shape for CUDA execution
//
#define j_block_sz (32)
#define i_block_sz (block_size / j_block_sz)

#define POLY_GEMM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA \
  j_block_sz, i_block_sz

#define POLY_GEMM_THREADS_PER_BLOCK_CUDA \
  dim3 nthreads_per_block(POLY_GEMM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA, 1);

#define POLY_GEMM_NBLOCKS_CUDA \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, j_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, i_block_sz)), \
               static_cast<size_t>(1));


template < size_t j_block_size, size_t i_block_size >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_gemm(Real_ptr C, Real_ptr A, Real_ptr B,
                          Real_type alpha, Real_type beta,
                          Index_type ni, Index_type nj, Index_type nk)
{
  Index_type i = blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = blockIdx.x * j_block_size + threadIdx.x;

  if ( i < ni && j < nj ) {
    POLYBENCH_GEMM_BODY1;
    POLYBENCH_GEMM_BODY2;
    for (Index_type k = 0; k < nk; ++k ) {
      POLYBENCH_GEMM_BODY3;
    }
    POLYBENCH_GEMM_BODY4;
  }
}

template < size_t j_block_size, size_t i_block_size, typename Lambda >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_gemm_lam(Index_type ni, Index_type nj,
                              Lambda body)
{
  Index_type i = blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = blockIdx.x * j_block_size + threadIdx.x;

  if ( i < ni && j < nj ) {
    body(i, j);
  }
}


template < size_t block_size >
void POLYBENCH_GEMM::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  POLYBENCH_GEMM_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      POLY_GEMM_THREADS_PER_BLOCK_CUDA;
      POLY_GEMM_NBLOCKS_CUDA;
      constexpr size_t shmem = 0;

      poly_gemm<POLY_GEMM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
               <<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(C, A, B,
                                                 alpha, beta,
                                                 ni, nj, nk);
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      POLY_GEMM_THREADS_PER_BLOCK_CUDA;
      POLY_GEMM_NBLOCKS_CUDA;
      constexpr size_t shmem = 0;

      poly_gemm_lam<POLY_GEMM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
                   <<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(ni, nj,
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

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_GEMM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixedAsync<i_block_sz * j_block_sz,
          RAJA::statement::For<0, RAJA::cuda_global_size_y_direct<i_block_sz>,   // i
            RAJA::statement::For<1, RAJA::cuda_global_size_x_direct<j_block_sz>, // j
              RAJA::statement::Lambda<0, RAJA::Params<0>>,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>>,
              RAJA::statement::For<2, RAJA::seq_exec,           // k
                RAJA::statement::Lambda<2, RAJA::Segs<0,1,2>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<3, RAJA::Segs<0,1>, RAJA::Params<0>>
            >
          >
        >
      >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param_resource<EXEC_POL>(

          RAJA::make_tuple( RAJA::RangeSegment{0, ni},
                            RAJA::RangeSegment{0, nj},
                            RAJA::RangeSegment{0, nk} ),
          RAJA::tuple<Real_type>{0.0},   // variable for dot
          res,

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

  } else {
      getCout() << "\n  POLYBENCH_GEMM : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_GEMM, Cuda)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA

