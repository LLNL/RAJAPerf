//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ADI.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

#define POLYBENCH_ADI_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(U, m_U, m_n * m_n); \
  allocAndInitHipDeviceData(V, m_V, m_n * m_n); \
  allocAndInitHipDeviceData(P, m_P, m_n * m_n); \
  allocAndInitHipDeviceData(Q, m_Q, m_n * m_n);


#define POLYBENCH_ADI_TEARDOWN_HIP \
  getHipDeviceData(m_U, U, m_n * m_n); \
  deallocHipDeviceData(U); \
  deallocHipDeviceData(V); \
  deallocHipDeviceData(P); \
  deallocHipDeviceData(Q);


template < size_t block_size >
__launch_bounds__(block_size)
__global__ void adi1(const Index_type n,
                     const Real_type a, const Real_type b, const Real_type c,
                     const Real_type d, const Real_type f,
                     Real_ptr P, Real_ptr Q, Real_ptr U, Real_ptr V)
{
  Index_type i = 1 + blockIdx.x * block_size + threadIdx.x;
  if (i < n-1) {
    POLYBENCH_ADI_BODY2;
    for (Index_type j = 1; j < n-1; ++j) {
       POLYBENCH_ADI_BODY3;
    }
    POLYBENCH_ADI_BODY4;
    for (Index_type k = n-2; k >= 1; --k) {
       POLYBENCH_ADI_BODY5;
    }
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void adi2(const Index_type n,
                     const Real_type a, const Real_type c, const Real_type d,
                     const Real_type e, const Real_type f,
                     Real_ptr P, Real_ptr Q, Real_ptr U, Real_ptr V)
{
  Index_type i = 1 + blockIdx.x * block_size + threadIdx.x;
  if (i < n-1) {
    POLYBENCH_ADI_BODY6;
    for (Index_type j = 1; j < n-1; ++j) {
      POLYBENCH_ADI_BODY7;
    }
    POLYBENCH_ADI_BODY8;
    for (Index_type k = n-2; k >= 1; --k) {
      POLYBENCH_ADI_BODY9;
    }
  }
}

template < size_t block_size, typename Lambda >
__launch_bounds__(block_size)
__global__ void adi_lam(const Index_type n,
                        Lambda body)
{
  Index_type i = 1 + blockIdx.x * block_size + threadIdx.x;
  if (i < n-1) {
    body(i);
  }
}


template < size_t block_size >
void POLYBENCH_ADI::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_ADI_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_ADI_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 1; t <= tsteps; ++t) {

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(n-2, block_size);

        hipLaunchKernelGGL((adi1<block_size>),
                           dim3(grid_size), dim3(block_size), 0, 0,
                           n,
                           a, b, c, d, f,
                           P, Q, U, V);
        hipErrchk( hipGetLastError() );

        hipLaunchKernelGGL((adi2<block_size>),
                           dim3(grid_size), dim3(block_size), 0, 0,
                           n,
                           a, c, d, e, f,
                           P, Q, U, V);
        hipErrchk( hipGetLastError() );

      }  // tstep loop

    }
    stopTimer();

    POLYBENCH_ADI_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    POLYBENCH_ADI_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 1; t <= tsteps; ++t) {

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(n-2, block_size);

        auto adi1_lamda = [=] __device__ (Index_type i) {
          POLYBENCH_ADI_BODY2;
          for (Index_type j = 1; j < n-1; ++j) {
             POLYBENCH_ADI_BODY3;
          }
          POLYBENCH_ADI_BODY4;
          for (Index_type k = n-2; k >= 1; --k) {
             POLYBENCH_ADI_BODY5;
          }
        };

        hipLaunchKernelGGL((adi_lam<block_size, decltype(adi1_lamda)>),
                           dim3(grid_size), dim3(block_size), 0, 0,
                           n, adi1_lamda);
        hipErrchk( hipGetLastError() );

        auto adi2_lamda = [=] __device__ (Index_type i) {
          POLYBENCH_ADI_BODY6;
          for (Index_type j = 1; j < n-1; ++j) {
            POLYBENCH_ADI_BODY7;
          }
          POLYBENCH_ADI_BODY8;
          for (Index_type k = n-2; k >= 1; --k) {
            POLYBENCH_ADI_BODY9;
          }
        };

        hipLaunchKernelGGL((adi_lam<block_size, decltype(adi2_lamda)>),
                           dim3(grid_size), dim3(block_size), 0, 0,
                           n, adi2_lamda);
        hipErrchk( hipGetLastError() );

      }  // tstep loop

    }
    stopTimer();

    POLYBENCH_ADI_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_ADI_DATA_SETUP_HIP;

    POLYBENCH_ADI_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<block_size,
          RAJA::statement::Tile<0, RAJA::tile_fixed<block_size>,
                                   RAJA::hip_block_x_direct,
            RAJA::statement::For<0, RAJA::hip_thread_x_direct,
              RAJA::statement::Lambda<0, RAJA::Segs<0>>,
              RAJA::statement::For<1, RAJA::seq_exec,
                RAJA::statement::Lambda<1, RAJA::Segs<0,1>>
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0>>,
              RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<3, RAJA::Segs<0,2>>
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 1; t <= tsteps; ++t) {

        RAJA::kernel<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                           RAJA::RangeSegment{1, n-1},
                           RAJA::RangeStrideSegment{n-2, 0, -1}),

          [=] __device__ (Index_type i) {
            POLYBENCH_ADI_BODY2_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_ADI_BODY3_RAJA;
          },
          [=] __device__ (Index_type i) {
            POLYBENCH_ADI_BODY4_RAJA;
          },
          [=] __device__ (Index_type i, Index_type k) {
            POLYBENCH_ADI_BODY5_RAJA;
          }
        );

        RAJA::kernel<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                           RAJA::RangeSegment{1, n-1},
                           RAJA::RangeStrideSegment{n-2, 0, -1}),

          [=] __device__ (Index_type i) {
            POLYBENCH_ADI_BODY6_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j) {
            POLYBENCH_ADI_BODY7_RAJA;
          },
          [=] __device__ (Index_type i) {
            POLYBENCH_ADI_BODY8_RAJA;
          },
          [=] __device__ (Index_type i, Index_type k) {
            POLYBENCH_ADI_BODY9_RAJA;
          }
        );

      }  // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_ADI_TEARDOWN_HIP

  } else {
      getCout() << "\n  POLYBENCH_ADI : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(POLYBENCH_ADI, Hip)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

