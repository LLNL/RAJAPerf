//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MEMSET.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

#define MEMSET_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, iend);

#define MEMSET_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x, x, iend); \
  deallocCudaDeviceData(x);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void memset(Real_ptr x, Real_type val,
                       Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if ( i < iend ) {
    MEMSET_BODY;
  }
}


void MEMSET::runCudaVariantLibrary(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMSET_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    MEMSET_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      cudaErrchk( cudaMemsetAsync(MEMSET_STD_ARGS, 0) );

    }
    stopTimer();

    MEMSET_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    MEMSET_DATA_SETUP_CUDA;

    camp::resources::Cuda res = camp::resources::Cuda::get_default();

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      res.memset(MEMSET_STD_ARGS);

    }
    stopTimer();

    MEMSET_DATA_TEARDOWN_CUDA;

  } else {

    getCout() << "\n  MEMSET : Unknown Cuda variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void MEMSET::runCudaVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMSET_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    MEMSET_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      memset<block_size><<<grid_size, block_size,
                  sizeof(Real_type)*block_size>>>( x,
                                                   val,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    MEMSET_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    MEMSET_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto memset_lambda = [=] __device__ (Index_type i) {
        MEMSET_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<block_size><<<grid_size, block_size>>>(
          ibegin, iend, memset_lambda );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    MEMSET_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    MEMSET_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          MEMSET_BODY;
      });

    }
    stopTimer();

    MEMSET_DATA_TEARDOWN_CUDA;

  } else {

    getCout() << "\n  MEMSET : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void MEMSET::runCudaVariant(VariantID vid, size_t tune_idx)
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

void MEMSET::setCudaTuningDefinitions(VariantID vid)
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
