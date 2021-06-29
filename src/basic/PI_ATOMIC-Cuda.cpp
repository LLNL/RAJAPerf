//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define PI_ATOMIC_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(pi, m_pi, 1);

#define PI_ATOMIC_DATA_TEARDOWN_CUDA \
  deallocCudaDeviceData(pi);

__global__ void pi_atomic(Real_ptr pi,
                          Real_type dx,
                          Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     double x = (double(i) + 0.5) * dx;
     RAJA::atomicAdd<RAJA::cuda_atomic>(pi, dx / (1.0 + x * x));
   }
}


void PI_ATOMIC::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  PI_ATOMIC_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    PI_ATOMIC_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initCudaDeviceData(pi, &m_pi_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      pi_atomic<<<grid_size, block_size>>>( pi, dx, iend );
      cudaErrchk( cudaGetLastError() );

      getCudaDeviceData(m_pi, pi, 1);
      *m_pi *= 4.0;

    }
    stopTimer();

    PI_ATOMIC_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    PI_ATOMIC_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initCudaDeviceData(pi, &m_pi_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<<<grid_size, block_size>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
          double x = (double(i) + 0.5) * dx;
          RAJA::atomicAdd<RAJA::cuda_atomic>(pi, dx / (1.0 + x * x));
      });
      cudaErrchk( cudaGetLastError() );

      getCudaDeviceData(m_pi, pi, 1);
      *m_pi *= 4.0;

    }
    stopTimer();

    PI_ATOMIC_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    PI_ATOMIC_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initCudaDeviceData(pi, &m_pi_init, 1);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          double x = (double(i) + 0.5) * dx;
          RAJA::atomicAdd<RAJA::cuda_atomic>(pi, dx / (1.0 + x * x));
      });

      getCudaDeviceData(m_pi, pi, 1);
      *m_pi *= 4.0;

    }
    stopTimer();

    PI_ATOMIC_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  PI_ATOMIC : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
