//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MEMCPY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

#define MEMCPY_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, iend); \
  allocAndInitCudaDeviceData(y, m_y, iend);

#define MEMCPY_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_y, y, iend); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void memcpy(Real_ptr x, Real_ptr y,
                       Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if ( i < iend ) {
    MEMCPY_BODY;
  }
}


void MEMCPY::runCudaVariantLibrary(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMCPY_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    MEMCPY_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      cudaErrchk( cudaMemcpyAsync(MEMCPY_STD_ARGS, cudaMemcpyDefault, 0) );

    }
    stopTimer();

    MEMCPY_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    MEMCPY_DATA_SETUP_CUDA;

    camp::resources::Cuda res = camp::resources::Cuda::get_default();

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      res.memcpy(MEMCPY_STD_ARGS);

    }
    stopTimer();

    MEMCPY_DATA_TEARDOWN_CUDA;

  } else {

    getCout() << "\n  MEMCPY : Unknown Cuda variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void MEMCPY::runCudaVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMCPY_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    MEMCPY_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      memcpy<block_size><<<grid_size, block_size>>>(
          x, y, iend );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    MEMCPY_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    MEMCPY_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto memcpy_lambda = [=] __device__ (Index_type i) {
        MEMCPY_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<block_size><<<grid_size, block_size>>>(
          ibegin, iend, memcpy_lambda );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    MEMCPY_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    MEMCPY_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          MEMCPY_BODY;
      });

    }
    stopTimer();

    MEMCPY_DATA_TEARDOWN_CUDA;

  } else {

    getCout() << "\n  MEMCPY : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void MEMCPY::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if (vid == Base_CUDA || vid == RAJA_CUDA) {

    if (tune_idx == t) {

      runCudaVariantLibrary(vid);

    }

    t += 1;

  }

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      if (tune_idx == t) {

        runCudaVariantBlock<block_size>(vid);

      }

      t += 1;

    }

  });
}

void MEMCPY::setCudaTuningDefinitions(VariantID vid)
{
  if (vid == Base_CUDA || vid == RAJA_CUDA) {
    addVariantTuningName(vid, "library");
  }

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      addVariantTuningName(vid, "block_"+std::to_string(block_size));

    }

  });
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
