//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_1D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

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
void POLYBENCH_JACOBI_1D::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  POLYBENCH_JACOBI_1D_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);
        constexpr size_t shmem = 0;

        poly_jacobi_1D_1<block_size><<<grid_size, block_size, shmem, res.get_stream()>>>(A, B, N);
        cudaErrchk( cudaGetLastError() );

        poly_jacobi_1D_2<block_size><<<grid_size, block_size, shmem, res.get_stream()>>>(A, B, N);
        cudaErrchk( cudaGetLastError() );

      }

    }
    stopTimer();

  } else if (vid == RAJA_CUDA) {

    using EXEC_POL = RAJA::cuda_exec<block_size, true /*async*/>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::forall<EXEC_POL> ( res, RAJA::RangeSegment{1, N-1},
          [=] __device__ (Index_type i) {
            POLYBENCH_JACOBI_1D_BODY1;
        });

        RAJA::forall<EXEC_POL> ( res, RAJA::RangeSegment{1, N-1},
          [=] __device__ (Index_type i) {
            POLYBENCH_JACOBI_1D_BODY2;
        });

      }

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_JACOBI_1D : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_JACOBI_1D, Cuda)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA

