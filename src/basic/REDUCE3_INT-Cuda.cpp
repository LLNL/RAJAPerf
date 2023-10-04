//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
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
#include <utility>


namespace rajaperf
{
namespace basic
{


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
void REDUCE3_INT::runCudaVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  REDUCE3_INT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    Int_ptr vmem_init;
    allocData(DataSpace::CudaPinned, vmem_init, 3);

    Int_ptr vmem;
    allocData(DataSpace::CudaDevice, vmem, 3);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      vmem_init[0] = m_vsum_init;
      vmem_init[1] = m_vmin_init;
      vmem_init[2] = m_vmax_init;
      cudaErrchk( cudaMemcpyAsync( vmem, vmem_init, 3*sizeof(Int_type),
                                   cudaMemcpyHostToDevice, res.get_stream() ) );

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 3*sizeof(Int_type)*block_size;
      reduce3int<block_size><<<grid_size, block_size,
                   shmem, res.get_stream()>>>(vec,
                                                    vmem + 0, m_vsum_init,
                                                    vmem + 1, m_vmin_init,
                                                    vmem + 2, m_vmax_init,
                                                    iend );
      cudaErrchk( cudaGetLastError() );

      Int_type lmem[3];
      cudaErrchk( cudaMemcpyAsync( &lmem[0], vmem, 3*sizeof(Int_type),
                                   cudaMemcpyDeviceToHost, res.get_stream() ) );
      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );
      m_vsum += lmem[0];
      m_vmin = RAJA_MIN(m_vmin, lmem[1]);
      m_vmax = RAJA_MAX(m_vmax, lmem[2]);

    }
    stopTimer();

    deallocData(DataSpace::CudaDevice, vmem);
    deallocData(DataSpace::CudaPinned, vmem_init);

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Int_type> vsum(m_vsum_init);
      RAJA::ReduceMin<RAJA::cuda_reduce, Int_type> vmin(m_vmin_init);
      RAJA::ReduceMax<RAJA::cuda_reduce, Int_type> vmax(m_vmax_init);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        REDUCE3_INT_BODY_RAJA;
      });

      m_vsum += static_cast<Int_type>(vsum.get());
      m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
      m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));

    }
    stopTimer();

  } else {
     getCout() << "\n  REDUCE3_INT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void REDUCE3_INT::runCudaVariantOccGS(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  REDUCE3_INT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    Int_ptr vmem_init;
    allocData(DataSpace::CudaPinned, vmem_init, 3);

    Int_ptr vmem;
    allocData(DataSpace::CudaDevice, vmem, 3);

    constexpr size_t shmem = 3*sizeof(Int_type)*block_size;
    const size_t max_grid_size = detail::getCudaOccupancyMaxBlocks(
        (reduce3int<block_size>), block_size, shmem);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      vmem_init[0] = m_vsum_init;
      vmem_init[1] = m_vmin_init;
      vmem_init[2] = m_vmax_init;
      cudaErrchk( cudaMemcpyAsync( vmem, vmem_init, 3*sizeof(Int_type),
                                   cudaMemcpyHostToDevice, res.get_stream() ) );

      const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t grid_size = std::min(normal_grid_size, max_grid_size);
      reduce3int<block_size><<<grid_size, block_size,
                               shmem, res.get_stream()>>>(vec,
                                                    vmem + 0, m_vsum_init,
                                                    vmem + 1, m_vmin_init,
                                                    vmem + 2, m_vmax_init,
                                                    iend );
      cudaErrchk( cudaGetLastError() );

      Int_type lmem[3];
      cudaErrchk( cudaMemcpyAsync( &lmem[0], vmem, 3*sizeof(Int_type),
                                   cudaMemcpyDeviceToHost, res.get_stream() ) );
      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );
      m_vsum += lmem[0];
      m_vmin = RAJA_MIN(m_vmin, lmem[1]);
      m_vmax = RAJA_MAX(m_vmax, lmem[2]);

    }
    stopTimer();

    deallocData(DataSpace::CudaDevice, vmem);
    deallocData(DataSpace::CudaPinned, vmem_init);

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Int_type> vsum(m_vsum_init);
      RAJA::ReduceMin<RAJA::cuda_reduce, Int_type> vmin(m_vmin_init);
      RAJA::ReduceMax<RAJA::cuda_reduce, Int_type> vmax(m_vmax_init);

      RAJA::forall< RAJA::cuda_exec_occ_calc<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        REDUCE3_INT_BODY_RAJA;
      });

      m_vsum += static_cast<Int_type>(vsum.get());
      m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
      m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));

    }
    stopTimer();

  } else {
     getCout() << "\n  REDUCE3_INT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

void REDUCE3_INT::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (tune_idx == t) {

          setBlockSize(block_size);
          runCudaVariantBlock<block_size>(vid);

        }

        t += 1;

        if (tune_idx == t) {

          setBlockSize(block_size);
          runCudaVariantOccGS<block_size>(vid);

        }

        t += 1;

      }

    });

  } else {

    getCout() << "\n  REDUCE3_INT : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void REDUCE3_INT::setCudaTuningDefinitions(VariantID vid)
{
  if ( vid == Base_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        addVariantTuningName(vid, "block_"+std::to_string(block_size));

        addVariantTuningName(vid, "occgs_"+std::to_string(block_size));

      }

    });

  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
