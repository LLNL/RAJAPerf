//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void pi_reduce(Real_type dx,
                          Real_ptr dpi, Real_type pi_init,
                          Index_type iend)
{
  extern __shared__ Real_type ppi[ ];

  Index_type i = blockIdx.x * block_size + threadIdx.x;

  ppi[ threadIdx.x ] = pi_init;
  for ( ; i < iend ; i += gridDim.x * block_size ) {
    double x = (double(i) + 0.5) * dx;
    ppi[ threadIdx.x ] += dx / (1.0 + x * x);
  }
  __syncthreads();

  for ( i = block_size / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      ppi[ threadIdx.x ] += ppi[ threadIdx.x + i ];
    }
     __syncthreads();
  }

#if 1 // serialized access to shared data;
  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::cuda_atomic>( dpi, ppi[ 0 ] );
  }
#else // this doesn't work due to data races
  if ( threadIdx.x == 0 ) {
    *dpi += ppi[ 0 ];
  }
#endif
}



template < size_t block_size >
void PI_REDUCE::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_REDUCE_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    Real_ptr dpi;
    allocAndInitCudaDeviceData(dpi, &m_pi_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initCudaDeviceData(dpi, &m_pi_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      pi_reduce<block_size><<<grid_size, block_size,
                  sizeof(Real_type)*block_size>>>( dx,
                                                   dpi, m_pi_init,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

      Real_type lpi;
      Real_ptr plpi = &lpi;
      getCudaDeviceData(plpi, dpi, 1);

      m_pi = 4.0 * lpi;

    }
    stopTimer();

    deallocCudaDeviceData(dpi);

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> pi(m_pi_init);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         PI_REDUCE_BODY;
       });

      m_pi = 4.0 * static_cast<Real_type>(pi.get());

    }
    stopTimer();

  } else {
     getCout() << "\n  PI_REDUCE : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(PI_REDUCE, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
