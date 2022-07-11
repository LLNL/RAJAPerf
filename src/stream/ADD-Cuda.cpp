//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ADD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

#define ADD_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(a, m_a, iend); \
  allocAndInitCudaDeviceData(b, m_b, iend); \
  allocAndInitCudaDeviceData(c, m_c, iend);

#define ADD_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_c, c, iend); \
  deallocCudaDeviceData(a); \
  deallocCudaDeviceData(b); \
  deallocCudaDeviceData(c);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void add(Real_ptr c, Real_ptr a, Real_ptr b,
                     Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    ADD_BODY;
  }
}


template < size_t block_size >
void ADD::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ADD_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    ADD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      add<block_size><<<grid_size, block_size>>>( c, a, b,
                                      iend );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    ADD_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    ADD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<block_size><<<grid_size, block_size>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
        ADD_BODY;
      });
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    ADD_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    ADD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        ADD_BODY;
      });

    }
    stopTimer();

    ADD_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  ADD : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(ADD, Cuda)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA

