  
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read RAJAPerf/LICENSE.
//
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

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;

#define POLYBENCH_JACOBI_1D_DATA_SETUP_CUDA \
  Real_ptr A; \
  Real_ptr B; \
\
  allocAndInitCudaDeviceData(A, m_Ainit, m_N); \
  allocAndInitCudaDeviceData(B, m_Binit, m_N);


#define POLYBENCH_JACOBI_1D_TEARDOWN_CUDA \
  getCudaDeviceData(m_A, A, m_N); \
  getCudaDeviceData(m_B, B, m_N); \
  deallocCudaDeviceData(A); \
  deallocCudaDeviceData(B);


__global__ void poly_jacobi_1D_1(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i > 0 && i < N-1) {
     POLYBENCH_JACOBI_1D_BODY1;
   }
}

__global__ void poly_jacobi_1D_2(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i > 0 && i < N-1) {
     POLYBENCH_JACOBI_1D_BODY2;
   }
}


void POLYBENCH_JACOBI_1D::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;
  const Index_type tsteps = m_tsteps;

  if ( vid == Base_CUDA ) {

    POLYBENCH_JACOBI_1D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);

        poly_jacobi_1D_1<<<grid_size, block_size>>>(A, B, N);

        poly_jacobi_1D_2<<<grid_size, block_size>>>(A, B, N);

      }

    }
    stopTimer();

    POLYBENCH_JACOBI_1D_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_JACOBI_1D_DATA_SETUP_CUDA;

    using EXEC_POL = RAJA::cuda_exec<block_size, true /*async*/>;

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

    POLYBENCH_JACOBI_1D_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_JACOBI_1D : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
  
