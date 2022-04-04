//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_1D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

#define POLYBENCH_JACOBI_1D_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(A, m_Ainit, m_N); \
  allocAndInitHipDeviceData(B, m_Binit, m_N);


#define POLYBENCH_JACOBI_1D_TEARDOWN_HIP \
  getHipDeviceData(m_A, A, m_N); \
  getHipDeviceData(m_B, B, m_N); \
  deallocHipDeviceData(A); \
  deallocHipDeviceData(B);


template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_jacobi_1D_1(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;

   if (i > 0 && i < N-1) {
     POLYBENCH_JACOBI_1D_BODY1;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_jacobi_1D_2(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;

   if (i > 0 && i < N-1) {
     POLYBENCH_JACOBI_1D_BODY2;
   }
}


template < size_t block_size >
void POLYBENCH_JACOBI_1D::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_JACOBI_1D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_JACOBI_1D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);

        hipLaunchKernelGGL((poly_jacobi_1D_1<block_size>), dim3(grid_size), dim3(block_size), 0, 0,
                                            A, B, N);
        hipErrchk( hipGetLastError() );

        hipLaunchKernelGGL((poly_jacobi_1D_2<block_size>), dim3(grid_size), dim3(block_size), 0, 0,
                                            A, B, N);
        hipErrchk( hipGetLastError() );

      }

    }
    stopTimer();

    POLYBENCH_JACOBI_1D_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_JACOBI_1D_DATA_SETUP_HIP;

    using EXEC_POL = RAJA::hip_exec<block_size, true /*async*/>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::forall<EXEC_POL> ( RAJA::RangeSegment{1, N-1},
          [=] __device__ (Index_type i) {
            POLYBENCH_JACOBI_1D_BODY1;
        });

        RAJA::forall<EXEC_POL> ( RAJA::RangeSegment{1, N-1},
          [=] __device__ (Index_type i) {
            POLYBENCH_JACOBI_1D_BODY2;
        });

      }

    }
    stopTimer();

    POLYBENCH_JACOBI_1D_TEARDOWN_HIP;

  } else {
      getCout() << "\n  POLYBENCH_JACOBI_1D : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(POLYBENCH_JACOBI_1D, Hip)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

