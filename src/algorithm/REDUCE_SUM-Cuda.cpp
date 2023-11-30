//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include "cub/device/device_reduce.cuh"
#include "cub/util_allocator.cuh"

#include <iostream>
#include <utility>


namespace rajaperf
{
namespace algorithm
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum(Real_ptr x, Real_ptr sum, Real_type sum_init,
                           Index_type iend)
{
  extern __shared__ Real_type psum[ ];

  Index_type i = blockIdx.x * block_size + threadIdx.x;

  psum[ threadIdx.x ] = sum_init;
  for ( ; i < iend ; i += gridDim.x * block_size ) {
    psum[ threadIdx.x ] += x[i];
  }
  __syncthreads();

  for ( i = block_size / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      psum[ threadIdx.x ] += psum[ threadIdx.x + i ];
    }
     __syncthreads();
  }

  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::cuda_atomic>( sum, psum[ 0 ] );
  }
}


void REDUCE_SUM::runCudaVariantCub(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    cudaStream_t stream = res.get_stream();

    int len = iend - ibegin;

    DataSpace rds = getReductionDataSpace(vid);
    DataSpace hrds = hostAccessibleDataSpace(rds);
    const bool separate_buffers = (hrds != rds);

    Real_ptr sum;
    allocData(rds, sum, 1);
    Real_ptr hsum = sum;
    if (separate_buffers) {
      allocData(hrds, hsum, 1);
    }

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cudaErrchk(::cub::DeviceReduce::Reduce(d_temp_storage,
                                           temp_storage_bytes,
                                           x+ibegin,
                                           sum,
                                           len,
                                           ::cub::Sum(),
                                           m_sum_init,
                                           stream));

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocData(DataSpace::CudaDevice, temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Run
      cudaErrchk(::cub::DeviceReduce::Reduce(d_temp_storage,
                                             temp_storage_bytes,
                                             x+ibegin,
                                             sum,
                                             len,
                                             ::cub::Sum(),
                                             m_sum_init,
                                             stream));

      if (separate_buffers) {
        cudaErrchk( cudaMemcpyAsync( hsum, sum, sizeof(Real_type),
                                   cudaMemcpyDeviceToHost, stream ) );
      }

      cudaErrchk(cudaStreamSynchronize(stream));
      m_sum = *hsum;

    }
    stopTimer();

    // Free temporary storage
    deallocData(DataSpace::CudaDevice, temp_storage);
    deallocData(rds, sum);
    if (separate_buffers) {
      deallocData(hrds, hsum);
    }

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void REDUCE_SUM::runCudaVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    DataSpace rds = getReductionDataSpace(vid);
    DataSpace hrds = hostAccessibleDataSpace(rds);
    const bool separate_buffers = (hrds != rds);

    Real_ptr sum;
    allocData(rds, sum, 1);
    Real_ptr hsum = sum;
    if (separate_buffers) {
      allocData(hrds, hsum, 1);
    }

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      if (separate_buffers) {
        *hsum = m_sum_init;
        cudaErrchk( cudaMemcpyAsync( sum, hsum, sizeof(Real_type),
                                     cudaMemcpyHostToDevice, res.get_stream() ) );
      } else {
        *sum = m_sum_init;
      }

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = sizeof(Real_type)*block_size;
      reduce_sum<block_size><<<grid_size, block_size,
                  shmem, res.get_stream()>>>( x,
                                                   sum, m_sum_init,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

      if (separate_buffers) {
        cudaErrchk( cudaMemcpyAsync( hsum, sum, sizeof(Real_type),
                                     cudaMemcpyDeviceToHost, res.get_stream() ) );
      }
      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );
      m_sum = *hsum;

    }
    stopTimer();

    deallocData(rds, sum);
    if (separate_buffers) {
      deallocData(hrds, hsum);
    }

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> sum(m_sum_init);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          REDUCE_SUM_BODY;
      });

      m_sum = sum.get();

    }
    stopTimer();

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void REDUCE_SUM::runCudaVariantOccGS(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    DataSpace rds = getReductionDataSpace(vid);
    DataSpace hrds = hostAccessibleDataSpace(rds);
    const bool separate_buffers = (hrds != rds);

    Real_ptr sum;
    allocData(rds, sum, 1);
    Real_ptr hsum = sum;
    if (separate_buffers) {
      allocData(hrds, hsum, 1);
    }

    constexpr size_t shmem = sizeof(Real_type)*block_size;
    const size_t max_grid_size = detail::getCudaOccupancyMaxBlocks(
        (reduce_sum<block_size>), block_size, shmem);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      if (separate_buffers) {
        *hsum = m_sum_init;
        cudaErrchk( cudaMemcpyAsync( sum, hsum, sizeof(Real_type),
                                     cudaMemcpyHostToDevice, res.get_stream() ) );
      } else {
        *sum = m_sum_init;
      }

      const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t grid_size = std::min(normal_grid_size, max_grid_size);
      reduce_sum<block_size><<<grid_size, block_size,
                               shmem, res.get_stream()>>>( x,
                                                   sum, m_sum_init,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

      if (separate_buffers) {
        cudaErrchk( cudaMemcpyAsync( hsum, sum, sizeof(Real_type),
                                     cudaMemcpyDeviceToHost, res.get_stream() ) );
      }
      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );
      m_sum = *hsum;

    }
    stopTimer();

    deallocData(rds, sum);
    if (separate_buffers) {
      deallocData(hrds, hsum);
    }

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> sum(m_sum_init);

      RAJA::forall< RAJA::cuda_exec_occ_calc<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          REDUCE_SUM_BODY;
      });

      m_sum = sum.get();

    }
    stopTimer();

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void REDUCE_SUM::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_CUDA ) {

    if (tune_idx == t) {

      runCudaVariantCub(vid);

    }

    t += 1;

  }

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

    getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void REDUCE_SUM::setCudaTuningDefinitions(VariantID vid)
{
  if ( vid == Base_CUDA ) {

    addVariantTuningName(vid, "cub");

  }

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

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
