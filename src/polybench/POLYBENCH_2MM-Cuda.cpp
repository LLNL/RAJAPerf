//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_2MM.hpp"

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
#define in_block_sz (32)
#define out_block_sz (block_size / in_block_sz)

#define POLY_2MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA \
  in_block_sz, out_block_sz

#define POLY_2MM_THREADS_PER_BLOCK_CUDA \
  dim3 nthreads_per_block(POLY_2MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA, 1);

#define POLY_2MM_1_NBLOCKS_CUDA \
  dim3 nblocks1(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, in_block_sz)), \
                static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, out_block_sz)), \
                static_cast<size_t>(1));

#define POLY_2MM_2_NBLOCKS_CUDA \
  dim3 nblocks2(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nl, in_block_sz)), \
                static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, out_block_sz)), \
                static_cast<size_t>(1));


template < size_t in_block_size, size_t out_block_size >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_2mm_1(Real_ptr tmp, Real_ptr A, Real_ptr B,
                           Real_type alpha,
                           Index_type ni, Index_type nj, Index_type nk)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type j = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && j < nj ) {
    POLYBENCH_2MM_BODY1;
    for (Index_type k=0; k < nk; ++k) {
      POLYBENCH_2MM_BODY2;
    }
    POLYBENCH_2MM_BODY3;
  }
}

template < size_t in_block_size, size_t out_block_size, typename Lambda >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_2mm_1_lam(Index_type ni, Index_type nj,
                               Lambda body)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type j = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && j < nj ) {
    body(i, j);
  }
}

template < size_t in_block_size, size_t out_block_size >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_2mm_2(Real_ptr tmp, Real_ptr C, Real_ptr D,
                           Real_type beta,
                           Index_type ni,  Index_type nl, Index_type nj)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type l = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && l < nl ) {
    POLYBENCH_2MM_BODY4;
    for (Index_type j=0; j < nj; ++j) {
      POLYBENCH_2MM_BODY5;
    }
    POLYBENCH_2MM_BODY6;
  }
}

template < size_t in_block_size, size_t out_block_size, typename Lambda >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_2mm_2_lam(Index_type ni,  Index_type nl,
                               Lambda body)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type l = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && l < nl ) {
    body(i, l);
  }
}


template < size_t block_size >
void POLYBENCH_2MM::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  POLYBENCH_2MM_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      POLY_2MM_THREADS_PER_BLOCK_CUDA;
      constexpr size_t shmem = 0;

      POLY_2MM_1_NBLOCKS_CUDA;
      poly_2mm_1<POLY_2MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
                <<<nblocks1, nthreads_per_block, shmem, res.get_stream()>>>(tmp, A, B, alpha,
                                                   ni, nj, nk);
      cudaErrchk( cudaGetLastError() );

      POLY_2MM_2_NBLOCKS_CUDA;
      poly_2mm_2<POLY_2MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
                <<<nblocks2, nthreads_per_block, shmem, res.get_stream()>>>(tmp, C, D, beta,
                                                   ni, nl, nj);
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

  } else if (vid == Lambda_CUDA) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      POLY_2MM_THREADS_PER_BLOCK_CUDA;
      constexpr size_t shmem = 0;

      POLY_2MM_1_NBLOCKS_CUDA;
      poly_2mm_1_lam<POLY_2MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
                    <<<nblocks1, nthreads_per_block, shmem, res.get_stream()>>>(ni, nj,
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_2MM_BODY1;
          for (Index_type k=0; k < nk; ++k) {
            POLYBENCH_2MM_BODY2;
          }
          POLYBENCH_2MM_BODY3;
        }
      );
      cudaErrchk( cudaGetLastError() );

      POLY_2MM_2_NBLOCKS_CUDA;
      poly_2mm_2_lam<POLY_2MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
                    <<<nblocks2, nthreads_per_block, shmem, res.get_stream()>>>(ni, nl,
        [=] __device__ (Index_type i, Index_type l) {
          POLYBENCH_2MM_BODY4;
          for (Index_type j=0; j < nj; ++j) {
            POLYBENCH_2MM_BODY5;
          }
          POLYBENCH_2MM_BODY6;
        }
      );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_2MM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixedAsync<out_block_sz * in_block_sz,
          RAJA::statement::For<0, RAJA::cuda_global_size_y_direct<out_block_sz>,   // outer
            RAJA::statement::For<1, RAJA::cuda_global_size_x_direct<in_block_sz>, // inner
              RAJA::statement::Lambda<0, RAJA::Params<0>>,
              RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<1, RAJA::Segs<0,1,2>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0,1>, RAJA::Params<0>>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel_param_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nj},
                         RAJA::RangeSegment{0, nk}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] __device__ (Real_type &dot) {
          POLYBENCH_2MM_BODY1_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j, Index_type k,
                        Real_type &dot) {
          POLYBENCH_2MM_BODY2_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j,
                        Real_type &dot) {
          POLYBENCH_2MM_BODY3_RAJA;
        }
      );

      RAJA::kernel_param_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nl},
                         RAJA::RangeSegment{0, nj}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] __device__ (Real_type &dot) {
          POLYBENCH_2MM_BODY4_RAJA;
        },
        [=] __device__ (Index_type i, Index_type l, Index_type j,
                        Real_type &dot) {
          POLYBENCH_2MM_BODY5_RAJA;
        },
        [=] __device__ (Index_type i, Index_type l,
                        Real_type &dot) {
          POLYBENCH_2MM_BODY6_RAJA;
        }
      );

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_2MM : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_2MM, Cuda)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA

