//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
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


#define POLYBENCH_GEMVER_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(A, m_A, m_n * m_n); \
  allocAndInitHipDeviceData(u1, m_u1, m_n); \
  allocAndInitHipDeviceData(v1, m_v1, m_n); \
  allocAndInitHipDeviceData(u2, m_u2, m_n); \
  allocAndInitHipDeviceData(v2, m_v2, m_n); \
  allocAndInitHipDeviceData(w, m_w, m_n); \
  allocAndInitHipDeviceData(x, m_x, m_n); \
  allocAndInitHipDeviceData(y, m_y, m_n); \
  allocAndInitHipDeviceData(z, m_z, m_n);


#define POLYBENCH_GEMVER_TEARDOWN_HIP \
  getHipDeviceData(m_w, w, m_n); \
  deallocHipDeviceData(A); \
  deallocHipDeviceData(u1); \
  deallocHipDeviceData(v1); \
  deallocHipDeviceData(u2); \
  deallocHipDeviceData(v2); \
  deallocHipDeviceData(w); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(y); \
  deallocHipDeviceData(z);

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

  POLYBENCH_GEMVER_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_GEMVER_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      GEMVER_THREADS_PER_BLOCK_HIP;
      GEMVER_NBLOCKS_HIP;

      hipLaunchKernelGGL((poly_gemmver_1<GEMVER_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                         dim3(nblocks1), dim3(nthreads_per_block1), 0, 0,
                         A, u1, v1, u2, v2, n);
      hipErrchk( hipGetLastError() );

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_n, block_size);

      hipLaunchKernelGGL((poly_gemmver_2<block_size>),
                         dim3(grid_size), dim3(block_size), 0, 0,
                         A, x, y, beta, n);
      hipErrchk( hipGetLastError() );

      hipLaunchKernelGGL((poly_gemmver_3<block_size>),
                         dim3(grid_size), dim3(block_size), 0, 0,
                         x, z, n);
      hipErrchk( hipGetLastError() );

      hipLaunchKernelGGL((poly_gemmver_4<block_size>),
                         dim3(grid_size), dim3(block_size), 0, 0,
                         A, x, w, alpha, n);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    POLYBENCH_GEMVER_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    POLYBENCH_GEMVER_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      GEMVER_THREADS_PER_BLOCK_HIP;
      GEMVER_NBLOCKS_HIP;

      auto poly_gemmver_1_lambda = [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_GEMVER_BODY1;
      };

      hipLaunchKernelGGL((poly_gemmver_1_lam<GEMVER_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, decltype(poly_gemmver_1_lambda)>),
                         dim3(nblocks1), dim3(nthreads_per_block1), 0, 0,
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
        dim3(grid_size), dim3(block_size), 0, 0,
        n, poly_gemmver_2_lambda);
      hipErrchk( hipGetLastError() );

      auto poly_gemmver_3_lambda = [=] __device__ (Index_type i) {
          POLYBENCH_GEMVER_BODY5;
      };

      hipLaunchKernelGGL((poly_gemmver_234_lam<block_size, decltype(poly_gemmver_3_lambda)>),
        dim3(grid_size), dim3(block_size), 0, 0,
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
        dim3(grid_size), dim3(block_size), 0, 0,
        n, poly_gemmver_4_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    POLYBENCH_GEMVER_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_GEMVER_DATA_SETUP_HIP;

    POLYBENCH_GEMVER_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<i_block_sz * j_block_sz,
          RAJA::statement::Tile<0, RAJA::tile_fixed<i_block_sz>,
                                   RAJA::hip_block_y_direct,
            RAJA::statement::Tile<1, RAJA::tile_fixed<j_block_sz>,
                                     RAJA::hip_block_x_direct,
              RAJA::statement::For<0, RAJA::hip_thread_y_direct,   // i
                RAJA::statement::For<1, RAJA::hip_thread_x_direct, // j
                  RAJA::statement::Lambda<0>
                >
              >
            >
          >
        >
      >;

    using EXEC_POL24 =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<block_size,
          RAJA::statement::Tile<0, RAJA::tile_fixed<block_size>,
                                   RAJA::hip_block_x_direct,
            RAJA::statement::For<0, RAJA::hip_thread_x_direct,   // i
              RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
              RAJA::statement::For<1, RAJA::seq_exec,            // j
                RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
            >
          >
        >
      >;

    using EXEC_POL3 = RAJA::hip_exec<block_size, true /*async*/>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL1>( RAJA::make_tuple(RAJA::RangeSegment{0, n},
                                                RAJA::RangeSegment{0, n}),
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_GEMVER_BODY1_RAJA;
        }
      );

      RAJA::kernel_param<EXEC_POL24>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        RAJA::tuple<Real_type>{0.0},

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

      RAJA::forall<EXEC_POL3> (RAJA::RangeSegment{0, n},
        [=] __device__ (Index_type i) {
          POLYBENCH_GEMVER_BODY5_RAJA;
        }
      );

      RAJA::kernel_param<EXEC_POL24>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        RAJA::tuple<Real_type>{0.0},

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

    POLYBENCH_GEMVER_TEARDOWN_HIP;

  } else {
      getCout() << "\n  POLYBENCH_GEMVER : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(POLYBENCH_GEMVER, Hip)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

