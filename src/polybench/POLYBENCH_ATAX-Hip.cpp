//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ATAX.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_atax_1(Real_ptr A, Real_ptr x, Real_ptr y, Real_ptr tmp,
                            Index_type N)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;

   if (i < N) {
     POLYBENCH_ATAX_BODY1;
     for (Index_type j = 0; j < N; ++j ) {
       POLYBENCH_ATAX_BODY2;
     }
     POLYBENCH_ATAX_BODY3;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_atax_2(Real_ptr A, Real_ptr tmp, Real_ptr y,
                            Index_type N)
{
   Index_type j = blockIdx.x * block_size + threadIdx.x;

   if (j < N) {
     POLYBENCH_ATAX_BODY4;
     for (Index_type i = 0; i < N; ++i ) {
       POLYBENCH_ATAX_BODY5;
     }
     POLYBENCH_ATAX_BODY6;
   }
}

template < size_t block_size, typename Lambda >
__launch_bounds__(block_size)
__global__ void poly_atax_lam(Index_type N,
                              Lambda body)
{
  Index_type ti = blockIdx.x * block_size + threadIdx.x;

  if (ti < N) {
    body(ti);
  }
}


template < size_t block_size >
void POLYBENCH_ATAX::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_ATAX_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);
      constexpr size_t shmem = 0;

      hipLaunchKernelGGL((poly_atax_1<block_size>),
                         dim3(grid_size), dim3(block_size), shmem, res.get_stream(),
                         A, x, y, tmp, N);
      hipErrchk( hipGetLastError() );

      hipLaunchKernelGGL((poly_atax_2<block_size>),
                         dim3(grid_size), dim3(block_size), shmem, res.get_stream(),
                         A, tmp, y, N);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);
      constexpr size_t shmem = 0;

      auto poly_atax_1_lambda = [=] __device__ (Index_type i) {
        POLYBENCH_ATAX_BODY1;
        for (Index_type j = 0; j < N; ++j ) {
          POLYBENCH_ATAX_BODY2;
        }
        POLYBENCH_ATAX_BODY3;
      };

      hipLaunchKernelGGL((poly_atax_lam<block_size, decltype(poly_atax_1_lambda)>),
        dim3(grid_size), dim3(block_size), shmem, res.get_stream(),
        N, poly_atax_1_lambda);
      hipErrchk( hipGetLastError() );

      auto poly_atax_2_lambda = [=] __device__ (Index_type j) {
        POLYBENCH_ATAX_BODY4;
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_ATAX_BODY5;
        }
        POLYBENCH_ATAX_BODY6;
      };

      hipLaunchKernelGGL((poly_atax_lam<block_size, decltype(poly_atax_2_lambda)>),
        dim3(grid_size), dim3(block_size), shmem, res.get_stream(),
        N, poly_atax_2_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    POLYBENCH_ATAX_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<block_size,
          RAJA::statement::For<0, RAJA::hip_global_size_x_direct<block_size>,
            RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
          >
        >
      >;

    using EXEC_POL2 =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<block_size,
          RAJA::statement::For<1, RAJA::hip_global_size_x_direct<block_size>,
            RAJA::statement::Lambda<0, RAJA::Segs<1>, RAJA::Params<0>>,
            RAJA::statement::For<0, RAJA::seq_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<1>, RAJA::Params<0>>
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel_param_resource<EXEC_POL1>(
        RAJA::make_tuple(RAJA::RangeSegment{0, N},
                         RAJA::RangeSegment{0, N}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] __device__ (Index_type i, Real_type &dot) {
          POLYBENCH_ATAX_BODY1_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_ATAX_BODY2_RAJA;
        },
        [=] __device__ (Index_type i, Real_type &dot) {
          POLYBENCH_ATAX_BODY3_RAJA;
        }

      );

      RAJA::kernel_param_resource<EXEC_POL2>(
        RAJA::make_tuple(RAJA::RangeSegment{0, N},
                         RAJA::RangeSegment{0, N}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] __device__ (Index_type j, Real_type &dot) {
          POLYBENCH_ATAX_BODY4_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j , Real_type &dot) {
          POLYBENCH_ATAX_BODY5_RAJA;
        },
        [=] __device__ (Index_type j, Real_type &dot) {
          POLYBENCH_ATAX_BODY6_RAJA;
        }

     );

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_ATAX : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_ATAX, Hip)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

