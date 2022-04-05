//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PLANCKIAN.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>
#include <cmath>

namespace rajaperf
{
namespace lcals
{

#define PLANCKIAN_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, iend); \
  allocAndInitCudaDeviceData(y, m_y, iend); \
  allocAndInitCudaDeviceData(u, m_u, iend); \
  allocAndInitCudaDeviceData(v, m_v, iend); \
  allocAndInitCudaDeviceData(w, m_w, iend);

#define PLANCKIAN_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_w, w, iend); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(u); \
  deallocCudaDeviceData(v); \
  deallocCudaDeviceData(w);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void planckian(Real_ptr x, Real_ptr y,
                          Real_ptr u, Real_ptr v, Real_ptr w,
                          Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     PLANCKIAN_BODY;
   }
}


template < size_t block_size >
void PLANCKIAN::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PLANCKIAN_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    PLANCKIAN_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       planckian<block_size><<<grid_size, block_size>>>( x, y,
                                             u, v, w,
                                             iend );
       cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    PLANCKIAN_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    PLANCKIAN_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         PLANCKIAN_BODY;
       });

    }
    stopTimer();

    PLANCKIAN_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  PLANCKIAN : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(PLANCKIAN, Cuda)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
