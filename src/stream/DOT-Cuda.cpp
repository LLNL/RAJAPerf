//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DOT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>


namespace rajaperf
{
namespace stream
{

#define DOT_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(a, m_a, iend); \
  allocAndInitCudaDeviceData(b, m_b, iend);

#define DOT_DATA_TEARDOWN_CUDA \
  deallocCudaDeviceData(a); \
  deallocCudaDeviceData(b);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void dot(Real_ptr a, Real_ptr b,
                    Real_ptr dprod, Real_type dprod_init,
                    Index_type iend)
{
  extern __shared__ Real_type pdot[ ];

  Index_type i = blockIdx.x * block_size + threadIdx.x;

  pdot[ threadIdx.x ] = dprod_init;
  for ( ; i < iend ; i += gridDim.x * block_size ) {
    pdot[ threadIdx.x ] += a[ i ] * b[i];
  }
  __syncthreads();

  for ( i = block_size / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      pdot[ threadIdx.x ] += pdot[ threadIdx.x + i ];
    }
     __syncthreads();
  }

#if 1 // serialized access to shared data;
  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::cuda_atomic>( dprod, pdot[ 0 ] );
  }
#else // this doesn't work due to data races
  if ( threadIdx.x == 0 ) {
    *dprod += pdot[ 0 ];
  }
#endif

}


template < size_t block_size >
void DOT::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DOT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    DOT_DATA_SETUP_CUDA;

    Real_ptr dprod;
    allocAndInitCudaDeviceData(dprod, &m_dot_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initCudaDeviceData(dprod, &m_dot_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      dot<block_size><<<grid_size, block_size, sizeof(Real_type)*block_size>>>(
          a, b, dprod, m_dot_init, iend );
      cudaErrchk( cudaGetLastError() );

      Real_type lprod;
      Real_ptr plprod = &lprod;
      getCudaDeviceData(plprod, dprod, 1);
      m_dot += lprod;

    }
    stopTimer();

    DOT_DATA_TEARDOWN_CUDA;

    deallocCudaDeviceData(dprod);

  } else if ( vid == RAJA_CUDA ) {

    DOT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> dot(m_dot_init);

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         DOT_BODY;
       });

       m_dot += static_cast<Real_type>(dot.get());

    }
    stopTimer();

    DOT_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  DOT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(DOT, Cuda)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
