//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define REDUCE3_INT_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(vec, m_vec, iend);

#define REDUCE3_INT_DATA_TEARDOWN_CUDA \
  deallocCudaDeviceData(vec);


template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce3int(Int_ptr vec,
                           Int_ptr vsum, Int_type vsum_init,
                           Int_ptr vmin, Int_type vmin_init,
                           Int_ptr vmax, Int_type vmax_init,
                           Index_type iend)
{
  extern __shared__ Int_type psum[ ];
  Int_type* pmin = (Int_type*)&psum[ 1 * block_size ];
  Int_type* pmax = (Int_type*)&psum[ 2 * block_size ];

  Index_type i = blockIdx.x * block_size + threadIdx.x;

  psum[ threadIdx.x ] = vsum_init;
  pmin[ threadIdx.x ] = vmin_init;
  pmax[ threadIdx.x ] = vmax_init;

  for ( ; i < iend ; i += gridDim.x * block_size ) {
    psum[ threadIdx.x ] += vec[ i ];
    pmin[ threadIdx.x ] = RAJA_MIN( pmin[ threadIdx.x ], vec[ i ] );
    pmax[ threadIdx.x ] = RAJA_MAX( pmax[ threadIdx.x ], vec[ i ] );
  }
  __syncthreads();

  for ( i = block_size / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      psum[ threadIdx.x ] += psum[ threadIdx.x + i ];
      pmin[ threadIdx.x ] = RAJA_MIN( pmin[ threadIdx.x ], pmin[ threadIdx.x + i ] );
      pmax[ threadIdx.x ] = RAJA_MAX( pmax[ threadIdx.x ], pmax[ threadIdx.x + i ] );
    }
     __syncthreads();
  }

#if 1 // serialized access to shared data;
  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::cuda_atomic>( vsum, psum[ 0 ] );
    RAJA::atomicMin<RAJA::cuda_atomic>( vmin, pmin[ 0 ] );
    RAJA::atomicMax<RAJA::cuda_atomic>( vmax, pmax[ 0 ] );
  }
#else // this doesn't work due to data races
  if ( threadIdx.x == 0 ) {
    *vsum += psum[ 0 ];
    *vmin = RAJA_MIN( *vmin, pmin[ 0 ] );
    *vmax = RAJA_MAX( *vmax, pmax[ 0 ] );
  }
#endif
}



template < size_t block_size >
void REDUCE3_INT::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE3_INT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    REDUCE3_INT_DATA_SETUP_CUDA;

    Int_ptr vmem_init;
    allocCudaPinnedData(vmem_init, 3);

    Int_ptr vmem;
    allocCudaDeviceData(vmem, 3);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      vmem_init[0] = m_vsum_init;
      vmem_init[1] = m_vmin_init;
      vmem_init[2] = m_vmax_init;
      cudaErrchk( cudaMemcpyAsync( vmem, vmem_init, 3*sizeof(Int_type),
                                   cudaMemcpyHostToDevice ) );

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      reduce3int<block_size><<<grid_size, block_size,
                   3*sizeof(Int_type)*block_size>>>(vec,
                                                    vmem + 0, m_vsum_init,
                                                    vmem + 1, m_vmin_init,
                                                    vmem + 2, m_vmax_init,
                                                    iend );
      cudaErrchk( cudaGetLastError() );

      Int_type lmem[3];
      Int_ptr plmem = &lmem[0];
      getCudaDeviceData(plmem, vmem, 3);
      m_vsum += lmem[0];
      m_vmin = RAJA_MIN(m_vmin, lmem[1]);
      m_vmax = RAJA_MAX(m_vmax, lmem[2]);

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_CUDA;

    deallocCudaDeviceData(vmem);
    deallocCudaPinnedData(vmem_init);

  } else if ( vid == RAJA_CUDA ) {

    REDUCE3_INT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Int_type> vsum(m_vsum_init);
      RAJA::ReduceMin<RAJA::cuda_reduce, Int_type> vmin(m_vmin_init);
      RAJA::ReduceMax<RAJA::cuda_reduce, Int_type> vmax(m_vmax_init);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        REDUCE3_INT_BODY_RAJA;
      });

      m_vsum += static_cast<Int_type>(vsum.get());
      m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
      m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  REDUCE3_INT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(REDUCE3_INT, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
