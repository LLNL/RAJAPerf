//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

namespace rajaperf
{
namespace polybench
{

//
// Define thread block size for CUDA execution
//
constexpr size_t i_block_sz = 8;
constexpr size_t j_block_sz = 32;

#define POLY_FLOYD_WARSHALL_THREADS_PER_BLOCK_CUDA \
  dim3 nthreads_per_block(j_block_sz, i_block_sz, 1);

#define POLY_FLOYD_WARSHALL_NBLOCKS_CUDA \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N, j_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N, i_block_sz)), \
               static_cast<size_t>(1));


#define POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(pin, m_pin, m_N * m_N); \
  allocAndInitCudaDeviceData(pout, m_pout, m_N * m_N);


#define POLYBENCH_FLOYD_WARSHALL_TEARDOWN_CUDA \
  getCudaDeviceData(m_pout, pout, m_N * m_N); \
  deallocCudaDeviceData(pin); \
  deallocCudaDeviceData(pout);


__global__ void poly_floyd_warshall(Real_ptr pout, Real_ptr pin,
                                    Index_type k,
                                    Index_type N)
{
  Index_type i = blockIdx.y * blockDim.y + threadIdx.y;
  Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i < N && j < N ) {
    POLYBENCH_FLOYD_WARSHALL_BODY;
  }
}

template< typename Lambda >
__global__ void poly_floyd_warshall_lam(Index_type N,
                                        Lambda body)
{
  Index_type i = blockIdx.y * blockDim.y + threadIdx.y;
  Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i < N && j < N ) {
    body(i, j);
  }
}


void POLYBENCH_FLOYD_WARSHALL::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_FLOYD_WARSHALL_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type k = 0; k < N; ++k) {

        POLY_FLOYD_WARSHALL_THREADS_PER_BLOCK_CUDA;
        POLY_FLOYD_WARSHALL_NBLOCKS_CUDA;

        poly_floyd_warshall<<<nblocks, nthreads_per_block>>>(pout, pin,
                                                             k, N);
        cudaErrchk( cudaGetLastError() );

      }

    }
    stopTimer();

    POLYBENCH_FLOYD_WARSHALL_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type k = 0; k < N; ++k) {

        POLY_FLOYD_WARSHALL_THREADS_PER_BLOCK_CUDA;
        POLY_FLOYD_WARSHALL_NBLOCKS_CUDA;

        poly_floyd_warshall_lam<<<nblocks, nthreads_per_block>>>(N,
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FLOYD_WARSHALL_BODY;
          }
        );

      }

    }
    stopTimer();

    POLYBENCH_FLOYD_WARSHALL_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_CUDA;

    POLYBENCH_FLOYD_WARSHALL_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::CudaKernelFixedAsync<i_block_sz * j_block_sz,
            RAJA::statement::Tile<1, RAJA::tile_fixed<i_block_sz>,
                                     RAJA::cuda_block_y_direct,
              RAJA::statement::Tile<2, RAJA::tile_fixed<j_block_sz>,
                                       RAJA::cuda_block_x_direct,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,   // i
                  RAJA::statement::For<2, RAJA::cuda_thread_x_direct, // j
                    RAJA::statement::Lambda<0>
                  >
                >
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                               RAJA::RangeSegment{0, N},
                                               RAJA::RangeSegment{0, N}),
        [=] __device__ (Index_type k, Index_type i, Index_type j) {
          POLYBENCH_FLOYD_WARSHALL_BODY_RAJA;
        }
      );

    }
    stopTimer();

    POLYBENCH_FLOYD_WARSHALL_TEARDOWN_CUDA;

  } else {
      getCout() << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA

