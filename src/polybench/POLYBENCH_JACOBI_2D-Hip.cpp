//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_2D.hpp"

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

#define JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP \
  j_block_sz, i_block_sz

#define JACOBI_2D_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block(JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, 1);

#define JACOBI_2D_NBLOCKS_HIP \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N-2, j_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N-2, i_block_sz)), \
               static_cast<size_t>(1));


#define POLYBENCH_JACOBI_2D_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(A, m_Ainit, m_N*m_N); \
  allocAndInitHipDeviceData(B, m_Binit, m_N*m_N);


#define POLYBENCH_JACOBI_2D_TEARDOWN_HIP \
  getHipDeviceData(m_A, A, m_N*m_N); \
  getHipDeviceData(m_B, B, m_N*m_N); \
  deallocHipDeviceData(A); \
  deallocHipDeviceData(B);


template < size_t j_block_size, size_t i_block_size >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_jacobi_2D_1(Real_ptr A, Real_ptr B, Index_type N)
{
  Index_type i = 1 + blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = 1 + blockIdx.x * j_block_size + threadIdx.x;

  if ( i < N-1 && j < N-1 ) {
    POLYBENCH_JACOBI_2D_BODY1;
  }
}

template < size_t j_block_size, size_t i_block_size >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_jacobi_2D_2(Real_ptr A, Real_ptr B, Index_type N)
{
  Index_type i = 1 + blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = 1 + blockIdx.x * j_block_size + threadIdx.x;

  if ( i < N-1 && j < N-1 ) {
    POLYBENCH_JACOBI_2D_BODY2;
  }
}

template < size_t j_block_size, size_t i_block_size, typename Lambda >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_jacobi_2D_lam(Index_type N, Lambda body)
{
  Index_type i = 1 + blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = 1 + blockIdx.x * j_block_size + threadIdx.x;

  if ( i < N-1 && j < N-1 ) {
    body(i, j);
  }
}


template < size_t block_size >
void POLYBENCH_JACOBI_2D::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_JACOBI_2D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        JACOBI_2D_THREADS_PER_BLOCK_HIP;
        JACOBI_2D_NBLOCKS_HIP;

        hipLaunchKernelGGL((poly_jacobi_2D_1<JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                           dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                           A, B, N);
        hipErrchk( hipGetLastError() );

        hipLaunchKernelGGL((poly_jacobi_2D_2<JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                           dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                           A, B, N);
        hipErrchk( hipGetLastError() );

      }

    }
    stopTimer();

    POLYBENCH_JACOBI_2D_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    POLYBENCH_JACOBI_2D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        JACOBI_2D_THREADS_PER_BLOCK_HIP;
        JACOBI_2D_NBLOCKS_HIP;

        auto poly_jacobi_2D_1_lambda =
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY1;
          };

        hipLaunchKernelGGL((poly_jacobi_2D_lam<JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, decltype(poly_jacobi_2D_1_lambda)>),
                           dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                           N, poly_jacobi_2D_1_lambda);
        hipErrchk( hipGetLastError() );

        auto poly_jacobi_2D_2_lambda =
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY2;
          };

        hipLaunchKernelGGL((poly_jacobi_2D_lam<JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, decltype(poly_jacobi_2D_2_lambda)>),
                           dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                           N, poly_jacobi_2D_2_lambda);
        hipErrchk( hipGetLastError() );

      }

    }
    stopTimer();

    POLYBENCH_JACOBI_2D_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_JACOBI_2D_DATA_SETUP_HIP;

    POLYBENCH_JACOBI_2D_VIEWS_RAJA;

    using EXEC_POL =
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

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::kernel<EXEC_POL>(RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                RAJA::RangeSegment{1, N-1}),
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY1_RAJA;
          }
        );

        RAJA::kernel<EXEC_POL>(RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                RAJA::RangeSegment{1, N-1}),
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY2_RAJA;
          }
        );

      }

    }
    stopTimer();

    POLYBENCH_JACOBI_2D_TEARDOWN_HIP;

  } else {
      getCout() << "\n  POLYBENCH_JACOBI_2D : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(POLYBENCH_JACOBI_2D, Hip)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

