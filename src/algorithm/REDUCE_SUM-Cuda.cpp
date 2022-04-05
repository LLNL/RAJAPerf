//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include "cub/device/device_reduce.cuh"
#include "cub/util_allocator.cuh"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

#define REDUCE_SUM_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, iend);

#define REDUCE_SUM_DATA_TEARDOWN_CUDA \
  deallocCudaDeviceData(x);


void REDUCE_SUM::runCudaVariantCub(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    cudaStream_t stream = 0;

    int len = iend - ibegin;

    Real_type* sum_storage;
    allocCudaPinnedData(sum_storage, 1);

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cudaErrchk(::cub::DeviceReduce::Reduce(d_temp_storage,
                                           temp_storage_bytes,
                                           x+ibegin,
                                           sum_storage,
                                           len,
                                           ::cub::Sum(),
                                           m_sum_init,
                                           stream));

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocCudaDeviceData(temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Run
      cudaErrchk(::cub::DeviceReduce::Reduce(d_temp_storage,
                                             temp_storage_bytes,
                                             x+ibegin,
                                             sum_storage,
                                             len,
                                             ::cub::Sum(),
                                             m_sum_init,
                                             stream));

      cudaErrchk(cudaStreamSynchronize(stream));
      m_sum = *sum_storage;

    }
    stopTimer();

    // Free temporary storage
    deallocCudaDeviceData(temp_storage);
    deallocCudaPinnedData(sum_storage);

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void REDUCE_SUM::runCudaVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> sum(m_sum_init);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          REDUCE_SUM_BODY;
      });

      m_sum = sum.get();

    }
    stopTimer();

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;
  }
}

void REDUCE_SUM::runCudaVariant(VariantID vid, size_t tune_idx)
{
  if ( vid == Base_CUDA ) {
    runCudaVariantCub(vid);
  } else if ( vid == RAJA_CUDA ) {
    size_t t = 0;
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        if (tune_idx == t) {
          runCudaVariantBlock<block_size>(vid);
        }
        t += 1;
      }
    });
  } else {
     getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;
  }
}

void REDUCE_SUM::setCudaTuningDefinitions(VariantID vid)
{
  if ( vid == Base_CUDA ) {
    addVariantTuningName(vid, "cub");
  } else if ( vid == RAJA_CUDA ) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        addVariantTuningName(vid, "block_"+std::to_string(block_size));
      }
    });
  }
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
