//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MULADDSUB.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define MULADDSUB_DATA_SETUP_CUDA \
  allocAndInitCudaData(out1, m_out1, iend); \
  allocAndInitCudaData(out2, m_out2, iend); \
  allocAndInitCudaData(out3, m_out3, iend); \
  allocAndInitCudaData(in1, m_in1, iend); \
  allocAndInitCudaData(in2, m_in2, iend);

#define MULADDSUB_DATA_TEARDOWN_CUDA \
  getCudaData(m_out1, out1, iend); \
  getCudaData(m_out2, out2, iend); \
  getCudaData(m_out3, out3, iend); \
  deallocCudaData(out1); \
  deallocCudaData(out2); \
  deallocCudaData(out3); \
  deallocCudaData(in1); \
  deallocCudaData(in2);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void muladdsub(Real_ptr out1, Real_ptr out2, Real_ptr out3,
                          Real_ptr in1, Real_ptr in2,
                          Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    MULADDSUB_BODY;
  }
}



template < size_t block_size >
void MULADDSUB::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MULADDSUB_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    MULADDSUB_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      muladdsub<block_size><<<grid_size, block_size>>>( out1, out2, out3, in1, in2,
                                            iend );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    MULADDSUB_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    MULADDSUB_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<block_size><<<grid_size, block_size>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
        MULADDSUB_BODY;
      });
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    MULADDSUB_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    MULADDSUB_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        MULADDSUB_BODY;
      });

    }
    stopTimer();

    MULADDSUB_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  MULADDSUB : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(MULADDSUB, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
