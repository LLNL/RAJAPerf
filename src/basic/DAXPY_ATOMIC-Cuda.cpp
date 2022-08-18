//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define DAXPY_ATOMIC_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, iend); \
  allocAndInitCudaDeviceData(y, m_y, iend);

#define DAXPY_ATOMIC_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_y, y, iend); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void daxpy_atomic(Real_ptr y, Real_ptr x,
                      Real_type a,
                      Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     DAXPY_ATOMIC_RAJA_BODY(RAJA::cuda_atomic);
   }
}


template < size_t block_size >
void DAXPY_ATOMIC::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DAXPY_ATOMIC_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    DAXPY_ATOMIC_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      daxpy_atomic<block_size><<<grid_size, block_size>>>( y, x, a,
                                        iend );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    DAXPY_ATOMIC_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    DAXPY_ATOMIC_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<block_size><<<grid_size, block_size>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
        DAXPY_ATOMIC_RAJA_BODY(RAJA::cuda_atomic);
      });
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    DAXPY_ATOMIC_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    DAXPY_ATOMIC_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        DAXPY_ATOMIC_RAJA_BODY(RAJA::cuda_atomic);
      });

    }
    stopTimer();

    DAXPY_ATOMIC_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  DAXPY_ATOMIC : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(DAXPY_ATOMIC, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
