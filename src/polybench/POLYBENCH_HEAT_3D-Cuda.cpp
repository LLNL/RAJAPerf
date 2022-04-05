//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_HEAT_3D.hpp"

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
#define k_block_sz (32)
#define j_block_sz (block_size / k_block_sz)
#define i_block_sz (1)

#define HEAT_3D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA \
  k_block_sz, j_block_sz, i_block_sz

#define HEAT_3D_THREADS_PER_BLOCK_CUDA \
  dim3 nthreads_per_block(HEAT_3D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA);

#define HEAT_3D_NBLOCKS_CUDA \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N-2, k_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N-2, j_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N-2, i_block_sz)));


#define POLYBENCH_HEAT_3D_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(A, m_Ainit, m_N*m_N*m_N); \
  allocAndInitCudaDeviceData(B, m_Binit, m_N*m_N*m_N); \
  static_assert(k_block_sz*j_block_sz*i_block_sz == block_size, "Invalid block_size");


#define POLYBENCH_HEAT_3D_TEARDOWN_CUDA \
  getCudaDeviceData(m_A, A, m_N*m_N*m_N); \
  getCudaDeviceData(m_B, B, m_N*m_N*m_N); \
  deallocCudaDeviceData(A); \
  deallocCudaDeviceData(B);


template < size_t k_block_size, size_t j_block_size, size_t i_block_size >
__launch_bounds__(k_block_size*j_block_size*i_block_size)
__global__ void poly_heat_3D_1(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = 1 + blockIdx.z;
   Index_type j = 1 + blockIdx.y * j_block_size + threadIdx.y;
   Index_type k = 1 + blockIdx.x * k_block_size + threadIdx.x;

   if (i < N-1 && j < N-1 && k < N-1) {
     POLYBENCH_HEAT_3D_BODY1;
   }
}

template < size_t k_block_size, size_t j_block_size, size_t i_block_size >
__launch_bounds__(k_block_size*j_block_size*i_block_size)
__global__ void poly_heat_3D_2(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = 1 + blockIdx.z;
   Index_type j = 1 + blockIdx.y * j_block_size + threadIdx.y;
   Index_type k = 1 + blockIdx.x * k_block_size + threadIdx.x;

   if (i < N-1 && j < N-1 && k < N-1) {
     POLYBENCH_HEAT_3D_BODY2;
   }
}

template< size_t k_block_size, size_t j_block_size, size_t i_block_size, typename Lambda >
__launch_bounds__(k_block_size*j_block_size*i_block_size)
__global__ void poly_heat_3D_lam(Index_type N, Lambda body)
{
   Index_type i = 1 + blockIdx.z;
   Index_type j = 1 + blockIdx.y * j_block_size + threadIdx.y;
   Index_type k = 1 + blockIdx.x * k_block_size + threadIdx.x;

   if (i < N-1 && j < N-1 && k < N-1) {
     body(i, j, k);
   }
}


template < size_t block_size >
void POLYBENCH_HEAT_3D::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_HEAT_3D_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    POLYBENCH_HEAT_3D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        HEAT_3D_THREADS_PER_BLOCK_CUDA;
        HEAT_3D_NBLOCKS_CUDA;

        poly_heat_3D_1<HEAT_3D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
            <<<nblocks, nthreads_per_block>>>(A, B, N);
        cudaErrchk( cudaGetLastError() );

        poly_heat_3D_2<HEAT_3D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
            <<<nblocks, nthreads_per_block>>>(A, B, N);
        cudaErrchk( cudaGetLastError() );

      }

    }
    stopTimer();

    POLYBENCH_HEAT_3D_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    POLYBENCH_HEAT_3D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        HEAT_3D_THREADS_PER_BLOCK_CUDA;
        HEAT_3D_NBLOCKS_CUDA;

        poly_heat_3D_lam<HEAT_3D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
            <<<nblocks, nthreads_per_block>>>(N,
          [=] __device__ (Index_type i, Index_type j, Index_type k) {
            POLYBENCH_HEAT_3D_BODY1;
          }
        );
        cudaErrchk( cudaGetLastError() );

        poly_heat_3D_lam<HEAT_3D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
            <<<nblocks, nthreads_per_block>>>(N,
          [=] __device__ (Index_type i, Index_type j, Index_type k) {
            POLYBENCH_HEAT_3D_BODY2;
          }
        );
        cudaErrchk( cudaGetLastError() );

      }

    }
    stopTimer();

    POLYBENCH_HEAT_3D_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_HEAT_3D_DATA_SETUP_CUDA;

    POLYBENCH_HEAT_3D_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixedAsync<j_block_sz * k_block_sz,
          RAJA::statement::Tile<1, RAJA::tile_fixed<j_block_sz>,
                                   RAJA::cuda_block_y_direct,
            RAJA::statement::Tile<2, RAJA::tile_fixed<k_block_sz>,
                                     RAJA::cuda_block_x_direct,
              RAJA::statement::For<0, RAJA::cuda_block_z_direct,      // i
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,   // j
                  RAJA::statement::For<2, RAJA::cuda_thread_x_direct, // k
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

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                 RAJA::RangeSegment{1, N-1},
                                                 RAJA::RangeSegment{1, N-1}),
          [=] __device__ (Index_type i, Index_type j, Index_type k) {
            POLYBENCH_HEAT_3D_BODY1_RAJA;
          }
        );

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                 RAJA::RangeSegment{1, N-1},
                                                 RAJA::RangeSegment{1, N-1}),
          [=] __device__ (Index_type i, Index_type j, Index_type k) {
            POLYBENCH_HEAT_3D_BODY2_RAJA;
          }
        );

      }

    }
    stopTimer();

    POLYBENCH_HEAT_3D_TEARDOWN_CUDA;

  } else {
      getCout() << "\n  POLYBENCH_HEAT_3D : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(POLYBENCH_HEAT_3D, Cuda)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
