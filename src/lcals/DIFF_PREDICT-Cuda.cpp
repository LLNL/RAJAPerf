//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DIFF_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

#define DIFF_PREDICT_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(px, m_px, m_array_length); \
  allocAndInitCudaDeviceData(cx, m_cx, m_array_length);

#define DIFF_PREDICT_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_px, px, m_array_length); \
  deallocCudaDeviceData(px); \
  deallocCudaDeviceData(cx);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void diff_predict(Real_ptr px, Real_ptr cx,
                             const Index_type offset,
                             Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     DIFF_PREDICT_BODY;
   }
}


template < size_t block_size >
void DIFF_PREDICT::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DIFF_PREDICT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    DIFF_PREDICT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       diff_predict<block_size><<<grid_size, block_size>>>( px, cx,
                                                offset,
                                                iend );
       cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    DIFF_PREDICT_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    DIFF_PREDICT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         DIFF_PREDICT_BODY;
       });

    }
    stopTimer();

    DIFF_PREDICT_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  DIFF_PREDICT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

void DIFF_PREDICT::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      if (tune_idx == t) {
        runCudaVariantImpl<block_size>(vid);
      }
      t += 1;
    }
  });
}

void DIFF_PREDICT::setCudaTuningDefinitions(VariantID vid)
{
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      addVariantTuningName(vid, "block_"+std::to_string(block_size));
    }
  });
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
