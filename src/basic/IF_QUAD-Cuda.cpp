//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "IF_QUAD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define IF_QUAD_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(a, m_a, iend); \
  allocAndInitCudaDeviceData(b, m_b, iend); \
  allocAndInitCudaDeviceData(c, m_c, iend); \
  allocAndInitCudaDeviceData(x1, m_x1, iend); \
  allocAndInitCudaDeviceData(x2, m_x2, iend);

#define IF_QUAD_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x1, x1, iend); \
  getCudaDeviceData(m_x2, x2, iend); \
  deallocCudaDeviceData(a); \
  deallocCudaDeviceData(b); \
  deallocCudaDeviceData(c); \
  deallocCudaDeviceData(x1); \
  deallocCudaDeviceData(x2);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void ifquad(Real_ptr x1, Real_ptr x2,
                       Real_ptr a, Real_ptr b, Real_ptr c,
                       Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    IF_QUAD_BODY;
  }
}



template < size_t block_size >
void IF_QUAD::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  IF_QUAD_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    IF_QUAD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      ifquad<block_size><<<grid_size, block_size>>>( x1, x2, a, b, c, iend );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    IF_QUAD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<block_size><<<grid_size, block_size>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
        IF_QUAD_BODY;
      });
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    IF_QUAD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        IF_QUAD_BODY;
      });

    }
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  IF_QUAD : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(IF_QUAD, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
