//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMVER.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

//
// Define thread block shape for Hip execution
//
#define j_block_sz (32)
#define i_block_sz (block_size / j_block_sz)

#define GEMVER_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP \
  j_block_sz, i_block_sz

#define GEMVER_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block1(GEMVER_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, 1);

#define GEMVER_NBLOCKS_HIP \
  dim3 nblocks1(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(n, j_block_sz)), \
                static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(n, i_block_sz)), \
                static_cast<size_t>(1));


template < size_t j_block_size, size_t i_block_size >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_gemmver_1(Real_ptr A,
                               Real_ptr u1, Real_ptr v1,
                               Real_ptr u2, Real_ptr v2,
                               Index_type n)
{
  Index_type i = blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = blockIdx.x * j_block_size + threadIdx.x;

  if (i < n && j < n) {
    POLYBENCH_GEMVER_BODY1;
  }
}

template < size_t j_block_size, size_t i_block_size, typename Lambda >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_gemmver_1_lam(Index_type n, Lambda body)
{
  Index_type i = blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = blockIdx.x * j_block_size + threadIdx.x;

  if (i < n && j < n) {
    body(i, j);
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_gemmver_2(Real_ptr A,
                               Real_ptr x, Real_ptr y,
                               Real_type beta,
                               Index_type n)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < n) {
    POLYBENCH_GEMVER_BODY2;
    for (Index_type j = 0; j < n; ++j) {
      POLYBENCH_GEMVER_BODY3;
    }
    POLYBENCH_GEMVER_BODY4;
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_gemmver_3(Real_ptr x, Real_ptr z,
                               Index_type n)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < n) {
    POLYBENCH_GEMVER_BODY5;
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_gemmver_4(Real_ptr A,
                               Real_ptr x, Real_ptr w,
                               Real_type alpha,
                               Index_type n)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < n) {
    POLYBENCH_GEMVER_BODY6;
    for (Index_type j = 0; j < n; ++j) {
      POLYBENCH_GEMVER_BODY7;
    }
    POLYBENCH_GEMVER_BODY8;
  }
}

template < size_t block_size, typename Lambda >
__launch_bounds__(block_size)
__global__ void poly_gemmver_234_lam(Index_type n, Lambda body)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < n) {
    body(i);
  }
}


template < size_t block_size >
void POLYBENCH_GEMVER::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_GEMVER_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      GEMVER_THREADS_PER_BLOCK_HIP;
      GEMVER_NBLOCKS_HIP;
      constexpr size_t shmem = 0;

      hipLaunchKernelGGL((poly_gemmver_1<GEMVER_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                         dim3(nblocks1), dim3(nthreads_per_block1), shmem, res.get_stream(),
                         A, u1, v1, u2, v2, n);
      hipErrchk( hipGetLastError() );

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_n, block_size);

      hipLaunchKernelGGL((poly_gemmver_2<block_size>),
                         dim3(grid_size), dim3(block_size), shmem, res.get_stream(),
                         A, x, y, beta, n);
      hipErrchk( hipGetLastError() );

      hipLaunchKernelGGL((poly_gemmver_3<block_size>),
                         dim3(grid_size), dim3(block_size), shmem, res.get_stream(),
                         x, z, n);
      hipErrchk( hipGetLastError() );

      hipLaunchKernelGGL((poly_gemmver_4<block_size>),
                         dim3(grid_size), dim3(block_size), shmem, res.get_stream(),
                         A, x, w, alpha, n);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      GEMVER_THREADS_PER_BLOCK_HIP;
      GEMVER_NBLOCKS_HIP;
      constexpr size_t shmem = 0;

      auto poly_gemmver_1_lambda = [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_GEMVER_BODY1;
      };

      hipLaunchKernelGGL((poly_gemmver_1_lam<GEMVER_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, decltype(poly_gemmver_1_lambda)>),
                         dim3(nblocks1), dim3(nthreads_per_block1), shmem, res.get_stream(),
                         n, poly_gemmver_1_lambda);
      hipErrchk( hipGetLastError() );

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(n, block_size);

      auto poly_gemmver_2_lambda = [=] __device__ (Index_type i) {
          POLYBENCH_GEMVER_BODY2;
          for (Index_type j = 0; j < n; ++j) {
            POLYBENCH_GEMVER_BODY3;
          }
          POLYBENCH_GEMVER_BODY4;
      };

      hipLaunchKernelGGL((poly_gemmver_234_lam<block_size, decltype(poly_gemmver_2_lambda)>),
        dim3(grid_size), dim3(block_size), shmem, res.get_stream(),
        n, poly_gemmver_2_lambda);
      hipErrchk( hipGetLastError() );

      auto poly_gemmver_3_lambda = [=] __device__ (Index_type i) {
          POLYBENCH_GEMVER_BODY5;
      };

      hipLaunchKernelGGL((poly_gemmver_234_lam<block_size, decltype(poly_gemmver_3_lambda)>),
        dim3(grid_size), dim3(block_size), shmem, res.get_stream(),
        n, poly_gemmver_3_lambda);
      hipErrchk( hipGetLastError() );

      auto poly_gemmver_4_lambda = [=] __device__ (Index_type i) {
          POLYBENCH_GEMVER_BODY6;
          for (Index_type j = 0; j < n; ++j) {
            POLYBENCH_GEMVER_BODY7;
          }
          POLYBENCH_GEMVER_BODY8;
      };

      hipLaunchKernelGGL((poly_gemmver_234_lam<block_size, decltype(poly_gemmver_4_lambda)>),
        dim3(grid_size), dim3(block_size), shmem, res.get_stream(),
        n, poly_gemmver_4_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    POLYBENCH_GEMVER_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<i_block_sz * j_block_sz,
          RAJA::statement::For<0, RAJA::hip_global_size_y_direct<i_block_sz>,   // i
            RAJA::statement::For<1, RAJA::hip_global_size_x_direct<j_block_sz>, // j
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

    using EXEC_POL24 =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<block_size,
          RAJA::statement::For<0, RAJA::hip_global_size_x_direct<block_size>,   // i
            RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
            RAJA::statement::For<1, RAJA::seq_exec,            // j
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
          >
        >
      >;

    using EXEC_POL3 = RAJA::hip_exec<block_size, true /*async*/>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel_resource<EXEC_POL1>( RAJA::make_tuple(RAJA::RangeSegment{0, n},
                                                RAJA::RangeSegment{0, n}),
                                        res,
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_GEMVER_BODY1_RAJA;
        }
      );

      RAJA::kernel_param_resource<EXEC_POL24>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] __device__ (Index_type /* i */, Real_type &dot) {
          POLYBENCH_GEMVER_BODY2_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_GEMVER_BODY3_RAJA;
        },
        [=] __device__ (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY4_RAJA;
        }
      );

      RAJA::forall<EXEC_POL3> ( res, RAJA::RangeSegment{0, n},
        [=] __device__ (Index_type i) {
          POLYBENCH_GEMVER_BODY5_RAJA;
        }
      );

      RAJA::kernel_param_resource<EXEC_POL24>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] __device__ (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY6_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_GEMVER_BODY7_RAJA;
        },
        [=] __device__ (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY8_RAJA;
        }
      );

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_GEMVER : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_GEMVER, Hip)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

