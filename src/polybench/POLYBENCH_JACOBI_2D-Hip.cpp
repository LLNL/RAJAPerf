//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
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

  auto res{getHipResource()};

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        JACOBI_2D_THREADS_PER_BLOCK_HIP;
        JACOBI_2D_NBLOCKS_HIP;
        constexpr size_t shmem = 0;

        hipLaunchKernelGGL((poly_jacobi_2D_1<JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                           dim3(nblocks), dim3(nthreads_per_block), shmem, res.get_stream(),
                           A, B, N);
        hipErrchk( hipGetLastError() );

        hipLaunchKernelGGL((poly_jacobi_2D_2<JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                           dim3(nblocks), dim3(nthreads_per_block), shmem, res.get_stream(),
                           A, B, N);
        hipErrchk( hipGetLastError() );

      }

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        JACOBI_2D_THREADS_PER_BLOCK_HIP;
        JACOBI_2D_NBLOCKS_HIP;
        constexpr size_t shmem = 0;

        auto poly_jacobi_2D_1_lambda =
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY1;
          };

        hipLaunchKernelGGL((poly_jacobi_2D_lam<JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, decltype(poly_jacobi_2D_1_lambda)>),
                           dim3(nblocks), dim3(nthreads_per_block), shmem, res.get_stream(),
                           N, poly_jacobi_2D_1_lambda);
        hipErrchk( hipGetLastError() );

        auto poly_jacobi_2D_2_lambda =
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY2;
          };

        hipLaunchKernelGGL((poly_jacobi_2D_lam<JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, decltype(poly_jacobi_2D_2_lambda)>),
                           dim3(nblocks), dim3(nthreads_per_block), shmem, res.get_stream(),
                           N, poly_jacobi_2D_2_lambda);
        hipErrchk( hipGetLastError() );

      }

    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    POLYBENCH_JACOBI_2D_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<i_block_sz * j_block_sz,
          RAJA::statement::For<0, RAJA::hip_global_size_y_direct<i_block_sz>,   // i
            RAJA::statement::For<1, RAJA::hip_global_size_x_direct<j_block_sz>, // j
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::kernel_resource<EXEC_POL>(RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                RAJA::RangeSegment{1, N-1}),
                                        res,
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY1_RAJA;
          }
        );

        RAJA::kernel_resource<EXEC_POL>(RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                RAJA::RangeSegment{1, N-1}),
                                        res,
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_JACOBI_2D_BODY2_RAJA;
          }
        );

      }

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_JACOBI_2D : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_JACOBI_2D, Hip)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

