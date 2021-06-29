//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_HEAT_3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

#define POLYBENCH_HEAT_3D_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(A, m_Ainit, m_N*m_N*m_N); \
  allocAndInitHipDeviceData(B, m_Binit, m_N*m_N*m_N);


#define POLYBENCH_HEAT_3D_TEARDOWN_HIP \
  getHipDeviceData(m_A, A, m_N*m_N*m_N); \
  getHipDeviceData(m_B, B, m_N*m_N*m_N); \
  deallocHipDeviceData(A); \
  deallocHipDeviceData(B);


__global__ void poly_heat_3D_1(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = 1 + blockIdx.y;
   Index_type j = 1 + blockIdx.z;
   Index_type k = 1 + threadIdx.x;

   if (i < N-1 && j < N-1 && k < N-1) {
     POLYBENCH_HEAT_3D_BODY1;
   }
}

__global__ void poly_heat_3D_2(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = 1 + blockIdx.y;
   Index_type j = 1 + blockIdx.z;
   Index_type k = 1 + threadIdx.x;

   if (i < N-1 && j < N-1 && k < N-1) {
     POLYBENCH_HEAT_3D_BODY2;
   }
}


void POLYBENCH_HEAT_3D::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_HEAT_3D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_HEAT_3D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        dim3 nblocks(1, N-2, N-2);
        dim3 nthreads_per_block(N-2, 1, 1);

        hipLaunchKernelGGL((poly_heat_3D_1),dim3(nblocks), dim3(nthreads_per_block),0,0,A, B, N);
        hipErrchk( hipGetLastError() );

        hipLaunchKernelGGL((poly_heat_3D_2),dim3(nblocks), dim3(nthreads_per_block),0,0,A, B, N);
        hipErrchk( hipGetLastError() );

      }

    }
    stopTimer();

    POLYBENCH_HEAT_3D_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    POLYBENCH_HEAT_3D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        dim3 nblocks(1, N-2, N-2);
        dim3 nthreads_per_block(N-2, 1, 1);

        auto poly_heat_3D_1_lambda = [=] __device__ (Index_type i, Index_type j, Index_type k) {

          POLYBENCH_HEAT_3D_BODY1;
        };

        auto kernel1 = lambda_hip_kernel<RAJA::hip_block_y_direct, RAJA::hip_block_z_direct, RAJA::hip_thread_x_direct, decltype(poly_heat_3D_1_lambda)>;
        hipLaunchKernelGGL(kernel1,
          nblocks, nthreads_per_block, 0, 0,
          1, N-1, 1, N-1, 1, N-1, poly_heat_3D_1_lambda);
        hipErrchk( hipGetLastError() );

        auto poly_heat_3D_2_lambda = [=] __device__ (Index_type i, Index_type j, Index_type k) {

          POLYBENCH_HEAT_3D_BODY2;
        };

        auto kernel2 = lambda_hip_kernel<RAJA::hip_block_y_direct, RAJA::hip_block_z_direct, RAJA::hip_thread_x_direct, decltype(poly_heat_3D_2_lambda)>;
        hipLaunchKernelGGL(kernel2,
          nblocks, nthreads_per_block, 0, 0,
          1, N-1, 1, N-1, 1, N-1, poly_heat_3D_2_lambda);
        hipErrchk( hipGetLastError() );

      }

    }
    stopTimer();

    POLYBENCH_HEAT_3D_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_HEAT_3D_DATA_SETUP_HIP;

    POLYBENCH_HEAT_3D_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelAsync<
          RAJA::statement::For<0, RAJA::hip_block_z_direct,
            RAJA::statement::For<1, RAJA::hip_block_y_direct,
              RAJA::statement::For<2, RAJA::hip_thread_x_direct,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >,
        RAJA::statement::HipKernelAsync<
          RAJA::statement::For<0, RAJA::hip_block_z_direct,
            RAJA::statement::For<1, RAJA::hip_block_y_direct,
              RAJA::statement::For<2, RAJA::hip_thread_x_direct,
                RAJA::statement::Lambda<1>
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                                                 RAJA::RangeSegment{1, N-1},
                                                 RAJA::RangeSegment{1, N-1}),
          [=] __device__ (Index_type i, Index_type j, Index_type k) {
            POLYBENCH_HEAT_3D_BODY1_RAJA;
          },
          [=] __device__ (Index_type i, Index_type j, Index_type k) {
            POLYBENCH_HEAT_3D_BODY2_RAJA;
          }
        );

      }

    }
    stopTimer();

    POLYBENCH_HEAT_3D_TEARDOWN_HIP;

  } else {
      std::cout << "\n  POLYBENCH_HEAT_3D : Unknown Hip variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
