//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  POLYBENCH_JACOBI_1D_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);
      constexpr size_t shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_1");
      RPlaunchCudaKernel( (poly_jacobi_1D_1<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          A, B, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_2");
      RPlaunchCudaKernel( (poly_jacobi_1D_2<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          A, B, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_2");

    }
    stopTimer();

  } else if (vid == RAJA_CUDA) {

    using EXEC_POL = RAJA::cuda_exec<block_size, true /*async*/>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_1");
      RAJA::forall<EXEC_POL> ( res, RAJA::RangeSegment{1, N-1},
        [=] __device__ (Index_type i) {
          POLYBENCH_JACOBI_1D_BODY1;
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_2");
      RAJA::forall<EXEC_POL> ( res, RAJA::RangeSegment{1, N-1},
        [=] __device__ (Index_type i) {
          POLYBENCH_JACOBI_1D_BODY2;
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_2");

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_JACOBI_1D : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_JACOBI_1D, Cuda, Base_CUDA, RAJA_CUDA)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
