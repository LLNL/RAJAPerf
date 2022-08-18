//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_MVT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

#define POLYBENCH_MVT_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x1, m_x1, N); \
  allocAndInitHipDeviceData(x2, m_x2, N); \
  allocAndInitHipDeviceData(y1, m_y1, N); \
  allocAndInitHipDeviceData(y2, m_y2, N); \
  allocAndInitHipDeviceData(A, m_A, N * N);


#define POLYBENCH_MVT_TEARDOWN_HIP \
  getHipDeviceData(m_x1, x1, N); \
  getHipDeviceData(m_x2, x2, N); \
  deallocHipDeviceData(x1); \
  deallocHipDeviceData(x2); \
  deallocHipDeviceData(y1); \
  deallocHipDeviceData(y2); \
  deallocHipDeviceData(A);


template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_mvt_1(Real_ptr A, Real_ptr x1, Real_ptr y1,
                           Index_type N)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;

   if (i < N) {
     POLYBENCH_MVT_BODY1;
     for (Index_type j = 0; j < N; ++j ) {
       POLYBENCH_MVT_BODY2;
     }
     POLYBENCH_MVT_BODY3;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_mvt_2(Real_ptr A, Real_ptr x2, Real_ptr y2,
                           Index_type N)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;

   if (i < N) {
     POLYBENCH_MVT_BODY4;
     for (Index_type j = 0; j < N; ++j ) {
       POLYBENCH_MVT_BODY5;
     }
     POLYBENCH_MVT_BODY6;
   }
}


template < size_t block_size >
void POLYBENCH_MVT::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_MVT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_MVT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);

      hipLaunchKernelGGL((poly_mvt_1<block_size>),
                         dim3(grid_size), dim3(block_size), 0, 0,
                         A, x1, y1, N);
      hipErrchk( hipGetLastError() );

      hipLaunchKernelGGL((poly_mvt_2<block_size>),
                         dim3(grid_size), dim3(block_size), 0, 0,
                         A, x2, y2, N);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    POLYBENCH_MVT_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_MVT_DATA_SETUP_HIP;

    POLYBENCH_MVT_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<block_size,
          RAJA::statement::Tile<0, RAJA::tile_fixed<block_size>,
                                   RAJA::hip_block_x_direct,
            RAJA::statement::For<0, RAJA::hip_thread_x_direct,  // i
              RAJA::statement::Lambda<0, RAJA::Params<0>>,
              RAJA::statement::For<1, RAJA::seq_exec,           // j
                RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, N},
                           RAJA::RangeSegment{0, N}),
          RAJA::tuple<Real_type>{0.0},

          [=] __device__ (Real_type &dot) {
            POLYBENCH_MVT_BODY1_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j, Real_type &dot) {
            POLYBENCH_MVT_BODY2_RAJA;
          },
          [=] __device__ (Index_type i, Real_type &dot) {
            POLYBENCH_MVT_BODY3_RAJA;
          }

        );

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, N},
                           RAJA::RangeSegment{0, N}),
          RAJA::tuple<Real_type>{0.0},

          [=] __device__ (Real_type &dot) {
            POLYBENCH_MVT_BODY4_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j, Real_type &dot) {
            POLYBENCH_MVT_BODY5_RAJA;
          },
          [=] __device__ (Index_type i, Real_type &dot) {
            POLYBENCH_MVT_BODY6_RAJA;
          }

        );

      }); // end sequential region (for single-source code)

    }
    stopTimer();

    POLYBENCH_MVT_TEARDOWN_HIP;

  } else {
      getCout() << "\n  POLYBENCH_MVT : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(POLYBENCH_MVT, Hip)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

