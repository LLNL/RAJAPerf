//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>
#include <utility>


namespace rajaperf
{
namespace basic
{

//
// Function used in TRAP_INT loop.
//
RAJA_INLINE
RAJA_DEVICE
Real_type trap_int_func(Real_type x,
                        Real_type y,
                        Real_type xp,
                        Real_type yp)
{
   Real_type denom = (x - xp)*(x - xp) + (y - yp)*(y - yp);
   denom = 1.0/sqrt(denom);
   return denom;
}


template < size_t block_size >
__launch_bounds__(block_size)
__global__ void trapint(Real_type x0, Real_type xp,
                        Real_type y, Real_type yp,
                        Real_type h,
                        Real_ptr sumx,
                        Index_type iend)
{
  extern __shared__ Real_type psumx[ ];

  Index_type i = blockIdx.x * block_size + threadIdx.x;

  psumx[ threadIdx.x ] = 0.0;
  for ( ; i < iend ; i += gridDim.x * block_size ) {
    Real_type x = x0 + i*h;
    Real_type val = trap_int_func(x, y, xp, yp);
    psumx[ threadIdx.x ] += val;
  }
  __syncthreads();

  for ( i = block_size / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      psumx[ threadIdx.x ] += psumx[ threadIdx.x + i ];
    }
     __syncthreads();
  }

#if 1 // serialized access to shared data;
  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::cuda_atomic>( sumx, psumx[ 0 ] );
  }
#else // this doesn't work due to data races
  if ( threadIdx.x == 0 ) {
    *sumx += psumx[ 0 ];
  }
#endif

}



template < size_t block_size >
void TRAP_INT::runCudaVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  TRAP_INT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    Real_ptr sumx;
    allocData(DataSpace::CudaDevice, sumx, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      cudaErrchk( cudaMemcpyAsync( sumx, &m_sumx_init, sizeof(Real_type),
                                   cudaMemcpyHostToDevice, res.get_stream() ) );

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = sizeof(Real_type)*block_size;
      trapint<block_size><<<grid_size, block_size,
                shmem, res.get_stream()>>>(x0, xp,
                                                y, yp,
                                                h,
                                                sumx,
                                                iend);
      cudaErrchk( cudaGetLastError() );

      Real_type lsumx;
      cudaErrchk( cudaMemcpyAsync( &lsumx, sumx, sizeof(Real_type),
                                   cudaMemcpyDeviceToHost, res.get_stream() ) );
      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );
      m_sumx += lsumx * h;

    }
    stopTimer();

    deallocData(DataSpace::CudaDevice, sumx);

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> sumx(m_sumx_init);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        TRAP_INT_BODY;
      });

      m_sumx += static_cast<Real_type>(sumx.get()) * h;

    }
    stopTimer();

  } else {
     getCout() << "\n  TRAP_INT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void TRAP_INT::runCudaVariantOccGS(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  TRAP_INT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    Real_ptr sumx;
    allocData(DataSpace::CudaDevice, sumx, 1);

    constexpr size_t shmem = sizeof(Real_type)*block_size;
    const size_t max_grid_size = detail::getCudaOccupancyMaxBlocks(
        (trapint<block_size>), block_size, shmem);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      cudaErrchk( cudaMemcpyAsync( sumx, &m_sumx_init, sizeof(Real_type),
                                   cudaMemcpyHostToDevice, res.get_stream() ) );

      const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t grid_size = std::min(normal_grid_size, max_grid_size);
      trapint<block_size><<<grid_size, block_size,
                            shmem, res.get_stream()>>>(x0, xp,
                                                y, yp,
                                                h,
                                                sumx,
                                                iend);
      cudaErrchk( cudaGetLastError() );

      Real_type lsumx;
      cudaErrchk( cudaMemcpyAsync( &lsumx, sumx, sizeof(Real_type),
                                   cudaMemcpyDeviceToHost, res.get_stream() ) );
      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );
      m_sumx += lsumx * h;

    }
    stopTimer();

    deallocData(DataSpace::CudaDevice, sumx);

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> sumx(m_sumx_init);

      RAJA::forall< RAJA::cuda_exec_occ_calc<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        TRAP_INT_BODY;
      });

      m_sumx += static_cast<Real_type>(sumx.get()) * h;

    }
    stopTimer();

  } else {
     getCout() << "\n  TRAP_INT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

void TRAP_INT::runCudaVariant(VariantID vid, size_t tune_idx)
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

    getCout() << "\n  TRAP_INT : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void TRAP_INT::setCudaTuningDefinitions(VariantID vid)
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
