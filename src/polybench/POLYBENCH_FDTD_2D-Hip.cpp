//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FDTD_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

  //
  // Define thread block size for Hip execution
  //
  const size_t block_size = 256;

  constexpr size_t j_block_sz = 32;
  constexpr size_t i_block_sz = 8;

#define FDTD_2D_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block234(j_block_sz, i_block_sz, 1);

#define FDTD_2D_NBLOCKS_HIP \
  dim3 nblocks234(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ny, j_block_sz)), \
                  static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nx, i_block_sz)), \
                  static_cast<size_t>(1));

#define POLYBENCH_FDTD_2D_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(hz, m_hz, m_nx * m_ny); \
  allocAndInitHipDeviceData(ex, m_ex, m_nx * m_ny); \
  allocAndInitHipDeviceData(ey, m_ey, m_nx * m_ny); \
  allocAndInitHipDeviceData(fict, m_fict, m_tsteps);


#define POLYBENCH_FDTD_2D_TEARDOWN_HIP \
  getHipDeviceData(m_hz, hz, m_nx * m_ny); \
  deallocHipDeviceData(ex); \
  deallocHipDeviceData(ey); \
  deallocHipDeviceData(fict);


__global__ void poly_fdtd2d_1(Real_ptr ey, Real_ptr fict,
                              Index_type ny, Index_type t)
{
   Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

   if (j < ny) {
     POLYBENCH_FDTD_2D_BODY1;
   }
}

template< typename Lambda >
__global__ void poly_fdtd2d_1_lam(Index_type ny, Lambda body)
{
  Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < ny) {
    body(j);
  }
}

__global__ void poly_fdtd2d_2(Real_ptr ey, Real_ptr hz,
                              Index_type nx, Index_type ny)
{
  Index_type i = blockIdx.y * blockDim.y + threadIdx.y;
  Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > 0 && i < nx && j < ny) {
    POLYBENCH_FDTD_2D_BODY2;
  }
}

template< typename Lambda >
__global__ void poly_fdtd2d_2_lam(Index_type nx, Index_type ny,
                                  Lambda body)
{
  Index_type i = blockIdx.y * blockDim.y + threadIdx.y;
  Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > 0 && i < nx && j < ny) {
    body(i, j);
  }
}

__global__ void poly_fdtd2d_3(Real_ptr ex, Real_ptr hz,
                              Index_type nx, Index_type ny)
{
  Index_type i = blockIdx.y * blockDim.y + threadIdx.y;
  Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nx && j > 0 && j < ny) {
    POLYBENCH_FDTD_2D_BODY3;
  }
}

template< typename Lambda >
__global__ void poly_fdtd2d_3_lam(Index_type nx, Index_type ny,
                                  Lambda body)
{
  Index_type i = blockIdx.y * blockDim.y + threadIdx.y;
  Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nx && j > 0 && j < ny) {
    body(i, j);
  }
}

__global__ void poly_fdtd2d_4(Real_ptr hz, Real_ptr ex, Real_ptr ey,
                              Index_type nx, Index_type ny)
{
  Index_type i = blockIdx.y * blockDim.y + threadIdx.y;
  Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nx-1 && j < ny-1) {
    POLYBENCH_FDTD_2D_BODY4;
  }
}

template< typename Lambda >
__global__ void poly_fdtd2d_4_lam(Index_type nx, Index_type ny,
                                  Lambda body)
{
  Index_type i = blockIdx.y * blockDim.y + threadIdx.y;
  Index_type j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nx-1 && j < ny-1) {
    body(i, j);
  }
}


void POLYBENCH_FDTD_2D::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_FDTD_2D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_FDTD_2D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (t = 0; t < tsteps; ++t) {

        const size_t grid_size1 = RAJA_DIVIDE_CEILING_INT(ny, block_size);
        hipLaunchKernelGGL((poly_fdtd2d_1),
                           dim3(grid_size1), dim3(block_size), 0, 0,
                           ey, fict, ny, t);
        hipErrchk( hipGetLastError() );

        FDTD_2D_THREADS_PER_BLOCK_HIP;
        FDTD_2D_NBLOCKS_HIP;

        hipLaunchKernelGGL((poly_fdtd2d_2),
                           dim3(nblocks234), dim3(nthreads_per_block234), 0, 0,
                           ey, hz, nx, ny);
        hipErrchk( hipGetLastError() );

        hipLaunchKernelGGL((poly_fdtd2d_3),
                           dim3(nblocks234), dim3(nthreads_per_block234), 0, 0,
                           ex, hz, nx, ny);
        hipErrchk( hipGetLastError() );

        hipLaunchKernelGGL((poly_fdtd2d_4),
                           dim3(nblocks234), dim3(nthreads_per_block234), 0, 0,
                           hz, ex, ey, nx, ny);
        hipErrchk( hipGetLastError() );

      } // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_FDTD_2D_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    POLYBENCH_FDTD_2D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (t = 0; t < tsteps; ++t) {

        const size_t grid_size1 = RAJA_DIVIDE_CEILING_INT(ny, block_size);

        auto poly_fdtd2d_1_lambda = [=] __device__ (Index_type j) {
          POLYBENCH_FDTD_2D_BODY1;
        };

        hipLaunchKernelGGL(poly_fdtd2d_1_lam<decltype(poly_fdtd2d_1_lambda)>,
          dim3(grid_size1), dim3(block_size), 0, 0,
          ny, poly_fdtd2d_1_lambda);
        hipErrchk( hipGetLastError() );

        FDTD_2D_THREADS_PER_BLOCK_HIP;
        FDTD_2D_NBLOCKS_HIP;

        auto poly_fdtd2d_2_lambda =
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY2;
          };

        hipLaunchKernelGGL((poly_fdtd2d_2_lam<decltype(poly_fdtd2d_2_lambda)>),
                           dim3(nblocks234), dim3(nthreads_per_block234), 0, 0,
                           nx, ny, poly_fdtd2d_2_lambda);
        hipErrchk( hipGetLastError() );

        auto poly_fdtd2d_3_lambda =
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY3;
          };

        hipLaunchKernelGGL((poly_fdtd2d_3_lam<decltype(poly_fdtd2d_3_lambda)>),
                           dim3(nblocks234), dim3(nthreads_per_block234), 0, 0,
                           nx, ny, poly_fdtd2d_3_lambda);
        hipErrchk( hipGetLastError() );

        auto poly_fdtd2d_4_lambda =
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY4;
          };

        hipLaunchKernelGGL((poly_fdtd2d_4_lam<decltype(poly_fdtd2d_4_lambda)>),
                           dim3(nblocks234), dim3(nthreads_per_block234), 0, 0,
                           nx, ny, poly_fdtd2d_4_lambda);
        hipErrchk( hipGetLastError() );

      } // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_FDTD_2D_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_FDTD_2D_DATA_SETUP_HIP;

    POLYBENCH_FDTD_2D_VIEWS_RAJA;

    using EXEC_POL1 = RAJA::hip_exec<block_size, true /*async*/>;

    using EXEC_POL234 =
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

      for (t = 0; t < tsteps; ++t) {

        RAJA::forall<EXEC_POL1>( RAJA::RangeSegment(0, ny),
         [=] __device__ (Index_type j) {
           POLYBENCH_FDTD_2D_BODY1_RAJA;
        });

        RAJA::kernel<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{1, nx},
                           RAJA::RangeSegment{0, ny}),
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY2_RAJA;
          }
        );

        RAJA::kernel<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nx},
                           RAJA::RangeSegment{1, ny}),
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY3_RAJA;
          }
        );

        RAJA::kernel<EXEC_POL234>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nx-1},
                           RAJA::RangeSegment{0, ny-1}),
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_FDTD_2D_BODY4_RAJA;
          }
        );

      }  // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_FDTD_2D_TEARDOWN_HIP;

  } else {
      getCout() << "\n  POLYBENCH_FDTD_2D : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

