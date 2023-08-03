//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block shape for CUDA execution
  //
#define i_block_sz (32)
#define j_block_sz (block_size / i_block_sz)
#define k_block_sz (1)

#define NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA \
  i_block_sz, j_block_sz, k_block_sz

#define NESTED_INIT_THREADS_PER_BLOCK_CUDA \
  dim3 nthreads_per_block(NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA); \
  static_assert(i_block_sz*j_block_sz*k_block_sz == block_size, "Invalid block_size");

#define NESTED_INIT_NBLOCKS_CUDA \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, i_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, j_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nk, k_block_sz)));


template< size_t i_block_size, size_t j_block_size, size_t k_block_size >
__launch_bounds__(i_block_size*j_block_size*k_block_size)
__global__ void nested_init(Real_ptr array,
                            Index_type ni, Index_type nj, Index_type nk)
{
  Index_type i = blockIdx.x * i_block_size + threadIdx.x;
  Index_type j = blockIdx.y * j_block_size + threadIdx.y;
  Index_type k = blockIdx.z;

  if ( i < ni && j < nj && k < nk ) {
    NESTED_INIT_BODY;
  }
}

template< size_t i_block_size, size_t j_block_size, size_t k_block_size, typename Lambda >
__launch_bounds__(i_block_size*j_block_size*k_block_size)
__global__ void nested_init_lam(Index_type ni, Index_type nj, Index_type nk,
                                Lambda body)
{
  Index_type i = blockIdx.x * i_block_size + threadIdx.x;
  Index_type j = blockIdx.y * j_block_size + threadIdx.y;
  Index_type k = blockIdx.z;

  if ( i < ni && j < nj && k < nk ) {
    body(i, j, k);
  }
}



template < size_t block_size >
void NESTED_INIT::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  NESTED_INIT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      NESTED_INIT_THREADS_PER_BLOCK_CUDA;
      NESTED_INIT_NBLOCKS_CUDA;
      constexpr size_t shmem = 0;

      nested_init<NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
                 <<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(array,
                                                   ni, nj, nk);
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      NESTED_INIT_THREADS_PER_BLOCK_CUDA;
      NESTED_INIT_NBLOCKS_CUDA;
      constexpr size_t shmem = 0;

      nested_init_lam<NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
                     <<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(ni, nj, nk,
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
          NESTED_INIT_BODY;
        }
      );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixedAsync<i_block_sz * j_block_sz,
          RAJA::statement::For<2, RAJA::cuda_block_z_direct,      // k
            RAJA::statement::For<1, RAJA::cuda_global_size_y_direct<j_block_sz>,   // j
              RAJA::statement::For<0, RAJA::cuda_global_size_x_direct<i_block_sz>, // i
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel_resource<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                               RAJA::RangeSegment(0, nj),
                                               RAJA::RangeSegment(0, nk)),
                                       res,
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
        NESTED_INIT_BODY;
      });

    }
    stopTimer();

  } else {
     getCout() << "\n  NESTED_INIT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(NESTED_INIT, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
