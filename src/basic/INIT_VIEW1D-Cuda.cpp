//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define INIT_VIEW1D_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(a, m_a, getActualProblemSize());

#define INIT_VIEW1D_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_a, a, getActualProblemSize()); \
  deallocCudaDeviceData(a);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void initview1d(Real_ptr a,
                           Real_type v,
                           const Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    INIT_VIEW1D_BODY;
  }
}



template < size_t block_size >
void INIT_VIEW1D::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INIT_VIEW1D_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    INIT_VIEW1D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      initview1d<block_size><<<grid_size, block_size>>>( a, v, iend );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    INIT_VIEW1D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<block_size><<<grid_size, block_size>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
        INIT_VIEW1D_BODY;
      });
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    INIT_VIEW1D_DATA_SETUP_CUDA;

    INIT_VIEW1D_VIEW_RAJA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        INIT_VIEW1D_BODY_RAJA;
      });

    }
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  INIT_VIEW1D : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(INIT_VIEW1D, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
