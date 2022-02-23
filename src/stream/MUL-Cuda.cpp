//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MUL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

#define MUL_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(b, m_b, iend); \
  allocAndInitCudaDeviceData(c, m_c, iend);

#define MUL_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_b, b, iend); \
  deallocCudaDeviceData(b); \
  deallocCudaDeviceData(c)

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void mul(Real_ptr b, Real_ptr c, Real_type alpha,
                    Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    MUL_BODY;
  }
}


template < size_t block_size >
void MUL::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MUL_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    MUL_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      mul<block_size><<<grid_size, block_size>>>( b, c, alpha,
                                      iend );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    MUL_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    MUL_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<block_size><<<grid_size, block_size>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
        MUL_BODY;
      });
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    MUL_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    MUL_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        MUL_BODY;
      });

    }
    stopTimer();

    MUL_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  MUL : Unknown Cuda variant id = " << vid << std::endl;
  }
}

void MUL::runCudaVariant(VariantID vid, size_t tid)
{
  size_t t = 0;
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      if (tid == t) {
        runCudaVariantImpl<block_size>(vid);
      }
      t += 1;
    }
  });
}

void MUL::setCudaTuningDefinitions(VariantID vid)
{
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      addVariantTuningName(vid, "block_"+std::to_string(block_size));
    }
  });
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
