//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void daxpy(Real_ptr y, Real_ptr x,
                      Real_type a,
                      Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     DAXPY_BODY;
   }
}


template < size_t block_size >
void DAXPY::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  DAXPY_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;
      daxpy<block_size><<<grid_size, block_size, shmem, res.get_stream()>>>( y, x, a,
                                        iend );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;
      lambda_cuda_forall<block_size><<<grid_size, block_size, shmem, res.get_stream()>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
        DAXPY_BODY;
      });
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        DAXPY_BODY;
      });

    }
    stopTimer();

  } else {
     getCout() << "\n  DAXPY : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(DAXPY, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
