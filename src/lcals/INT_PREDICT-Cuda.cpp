//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INT_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

#define INT_PREDICT_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(px, m_px, m_array_length);

#define INT_PREDICT_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_px, px, m_array_length); \
  deallocCudaDeviceData(px);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void int_predict(Real_ptr px,
                            Real_type dm22, Real_type dm23, Real_type dm24,
                            Real_type dm25, Real_type dm26, Real_type dm27,
                            Real_type dm28, Real_type c0,
                            const Index_type offset,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     INT_PREDICT_BODY;
   }
}


template < size_t block_size >
void INT_PREDICT::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INT_PREDICT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    INT_PREDICT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       int_predict<block_size><<<grid_size, block_size>>>( px,
                                               dm22, dm23, dm24, dm25,
                                               dm26, dm27, dm28, c0,
                                               offset,
                                               iend );
       cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    INT_PREDICT_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    INT_PREDICT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         INT_PREDICT_BODY;
       });

    }
    stopTimer();

    INT_PREDICT_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  INT_PREDICT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(INT_PREDICT, Cuda)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
