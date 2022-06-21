//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
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

namespace rajaperf
{
namespace algorithm
{

#define REDUCE_SUM_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, iend);

#define REDUCE_SUM_DATA_TEARDOWN_CUDA \
  deallocCudaDeviceData(x);

#define REDUCE_SUM_BODY_CUDA(atomicAdd) \
  RAJAPERF_REDUCE_1_CUDA(Real_type, REDUCE_SUM_VAL, dsum, sum_init, RAJAPERF_ADD_OP, atomicAdd)

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  REDUCE_SUM_BODY_CUDA(::atomicAdd)
}

constexpr size_t num_cuda_exp = 9;

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp0(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  extern __shared__ Real_type _shmem[];

  Real_type val = sum_init;

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  ::atomicAdd( dsum, val );
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp1(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  extern __shared__ Real_type _shmem[];

  Real_type val = sum_init;

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  for ( int i = RAJAPERF_CUDA_WARP / 2; i > 0; i /= 2 ) {
    val = (val + RAJAPERF_CUDA_SHFL_XOR(val, i));
  }

  if ( threadIdx.x % RAJAPERF_CUDA_WARP == 0 ) {
    ::atomicAdd( dsum, val );
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp2(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  extern __shared__ Real_type _shmem[];

  Real_type val = sum_init;

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  for ( int i = RAJAPERF_CUDA_WARP / 2; i > 0; i /= 2 ) {
    val = (val + RAJAPERF_CUDA_SHFL_XOR(val, i));
  }

  if ( threadIdx.x % RAJAPERF_CUDA_WARP == 0 ) {
    ::atomicAdd( &dsum[threadIdx.x / RAJAPERF_CUDA_WARP], val );
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp3(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  extern __shared__ Real_type _shmem[];

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    Real_type val = sum_init;
    if ( i < iend ) {
      val = x[i];
    }

    _shmem[ threadIdx.x ] = val;
  }
  __syncthreads();

  for ( int i = block_size / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      _shmem[ threadIdx.x ] = (_shmem[ threadIdx.x ] + _shmem[ threadIdx.x + i ]);
    }
     __syncthreads();
  }

  if ( threadIdx.x == 0 ) {
    ::atomicAdd( dsum, _shmem[ 0 ] );
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp4(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  extern __shared__ Real_type _shmem[];

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    Real_type val = sum_init;
    if ( i < iend ) {
      val = x[i];
    }

    _shmem[ threadIdx.x ] = val;
  }
  __syncthreads();

  for ( int i = block_size / 2; i >= RAJAPERF_CUDA_WARP; i /= 2 ) {
    if ( threadIdx.x < i ) {
      _shmem[ threadIdx.x ] = (_shmem[ threadIdx.x ] + _shmem[ threadIdx.x + i ]);
    }
     __syncthreads();
  }

  if ( threadIdx.x < RAJAPERF_CUDA_WARP) {
    Real_type val = _shmem[ threadIdx.x ];

    for ( int i = RAJAPERF_CUDA_WARP / 2; i > 0; i /= 2 ) {
      val = (val + RAJAPERF_CUDA_SHFL_XOR(val, i));
    }

    if ( threadIdx.x == 0 ) {
      ::atomicAdd( dsum, val );
    }
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp5(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  extern __shared__ Real_type _shmem[];

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    Real_type val = sum_init;
    if ( i < iend ) {
      val = x[i];
    }

    _shmem[ threadIdx.x ] = val;
  }
  __syncthreads();

  if ( threadIdx.x < RAJAPERF_CUDA_WARP) {

    Real_type val = sum_init;
    for ( int i = 0; i < block_size; i += RAJAPERF_CUDA_WARP ) {
      val = (val + _shmem[ threadIdx.x + i ]);
    }

    for ( int i = RAJAPERF_CUDA_WARP / 2; i > 0; i /= 2 ) {
      val = (val + RAJAPERF_CUDA_SHFL_XOR(val, i));
    }

    if ( threadIdx.x == 0 ) {
      ::atomicAdd( dsum, val );
    }
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp6(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  extern __shared__ Real_type _shmem[];

  Real_type val = sum_init;
  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  constexpr int wavefront_shrink_factor = block_size / RAJAPERF_CUDA_WARP;
  constexpr int wavefront_shrink_size = RAJAPERF_CUDA_WARP / wavefront_shrink_factor;
  const int wavefront_idx = threadIdx.x / RAJAPERF_CUDA_WARP;
  const int wavefront_thread_idx = threadIdx.x % RAJAPERF_CUDA_WARP;

  for ( int i = RAJAPERF_CUDA_WARP / 2; i >= wavefront_shrink_size; i /= 2 ) {
    val = (val + RAJAPERF_CUDA_SHFL_XOR(val, i));
  }

  if (wavefront_thread_idx < wavefront_shrink_size) {
    _shmem[ wavefront_thread_idx + wavefront_idx * wavefront_shrink_size ] = val;
  }

  __syncthreads();

  if ( threadIdx.x < RAJAPERF_CUDA_WARP) {
    Real_type val = _shmem[ threadIdx.x ];

    for ( int i = RAJAPERF_CUDA_WARP / 2; i > 0; i /= 2 ) {
      val = (val + RAJAPERF_CUDA_SHFL_XOR(val, i));
    }

    if ( threadIdx.x == 0 ) {
      ::atomicAdd( dsum, val );
    }
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp7(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  extern __shared__ Real_type _shmem[];

  Real_type val = sum_init;
  for ( Index_type i = blockIdx.x * block_size + threadIdx.x;
        i < iend ; i += gridDim.x * block_size ) {
    val = (val + x[i]);
  }

  constexpr int wavefront_shrink_factor = block_size / RAJAPERF_CUDA_WARP;
  constexpr int wavefront_shrink_size = RAJAPERF_CUDA_WARP / wavefront_shrink_factor;
  const int wavefront_idx = threadIdx.x / RAJAPERF_CUDA_WARP;
  const int wavefront_thread_idx = threadIdx.x % RAJAPERF_CUDA_WARP;

  for ( int i = RAJAPERF_CUDA_WARP / 2; i >= wavefront_shrink_size; i /= 2 ) {
    val = (val + RAJAPERF_CUDA_SHFL_XOR(val, i));
  }

  if (wavefront_thread_idx < wavefront_shrink_size) {
    _shmem[ wavefront_thread_idx + wavefront_idx * wavefront_shrink_size ] = val;
  }

  __syncthreads();

  if ( threadIdx.x < RAJAPERF_CUDA_WARP) {
    Real_type val = _shmem[ threadIdx.x ];

    for ( int i = RAJAPERF_CUDA_WARP / 2; i > 0; i /= 2 ) {
      val = (val + RAJAPERF_CUDA_SHFL_XOR(val, i));
    }

    if ( threadIdx.x == 0 ) {
      ::atomicAdd( dsum, val );
    }
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp8(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                                unsigned* dcnt, Real_ptr dtmp, Index_type iend)
{
  extern __shared__ Real_type _shmem[];

  Real_type val = sum_init;
  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  constexpr unsigned wavefront_shrink_factor = block_size / RAJAPERF_CUDA_WARP;
  constexpr unsigned wavefront_shrink_size = RAJAPERF_CUDA_WARP / wavefront_shrink_factor;
  const unsigned wavefront_idx = threadIdx.x / RAJAPERF_CUDA_WARP;
  const unsigned wavefront_thread_idx = threadIdx.x % RAJAPERF_CUDA_WARP;

  for ( unsigned i = RAJAPERF_CUDA_WARP / 2u; i >= wavefront_shrink_size; i /= 2u ) {
    val = (val + RAJAPERF_CUDA_SHFL_XOR(val, i));
  }

  if (wavefront_thread_idx < wavefront_shrink_size) {
    _shmem[ wavefront_thread_idx + wavefront_idx * wavefront_shrink_size ] = val;
  }

  __syncthreads();

  __shared__ unsigned reduction_block_idx;
  if ( threadIdx.x < RAJAPERF_CUDA_WARP) {
    Real_type val = _shmem[ threadIdx.x ];

    for ( unsigned i = RAJAPERF_CUDA_WARP / 2u; i > 0u; i /= 2u ) {
      val = (val + RAJAPERF_CUDA_SHFL_XOR(val, i));
    }

    if ( threadIdx.x == 0 ) {
      dtmp[blockIdx.x] = val;
      __threadfence();
      reduction_block_idx = ::atomicInc(dcnt, gridDim.x-1);
      __threadfence();
    }
  }

  __syncthreads();

  if (reduction_block_idx == gridDim.x-1) {

    val = sum_init;
    for ( unsigned i = threadIdx.x ; i < gridDim.x; i += block_size ) {
      val = (val + dtmp[i]);
    }

    for ( unsigned i = RAJAPERF_CUDA_WARP / 2u; i >= wavefront_shrink_size; i /= 2u ) {
      val = (val + RAJAPERF_CUDA_SHFL_XOR(val, i));
    }

    if (wavefront_thread_idx < wavefront_shrink_size) {
      _shmem[ wavefront_thread_idx + wavefront_idx * wavefront_shrink_size ] = val;
    }

    __syncthreads();

    if ( threadIdx.x < RAJAPERF_CUDA_WARP) {
      Real_type val = _shmem[ threadIdx.x ];

      for ( unsigned i = RAJAPERF_CUDA_WARP / 2u; i > 0u; i /= 2u ) {
        val = (val + RAJAPERF_CUDA_SHFL_XOR(val, i));
      }

      if ( threadIdx.x == 0 ) {
        dsum[0] = val;
      }
    }

  }

}


void REDUCE_SUM::runCudaVariantCub(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    cudaStream_t stream = 0;

    int len = iend - ibegin;

    Real_type* sum_storage;
    allocCudaPinnedData(sum_storage, 1);

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cudaErrchk(::cub::DeviceReduce::Reduce(d_temp_storage,
                                           temp_storage_bytes,
                                           x+ibegin,
                                           sum_storage,
                                           len,
                                           ::cub::Sum(),
                                           sum_init,
                                           stream));

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocCudaDeviceData(temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Run
      cudaErrchk(::cub::DeviceReduce::Reduce(d_temp_storage,
                                             temp_storage_bytes,
                                             x+ibegin,
                                             sum_storage,
                                             len,
                                             ::cub::Sum(),
                                             sum_init,
                                             stream));

      cudaErrchk(cudaStreamSynchronize(stream));
      m_sum = *sum_storage;

    }
    stopTimer();

    // Free temporary storage
    deallocCudaDeviceData(temp_storage);
    deallocCudaPinnedData(sum_storage);

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void REDUCE_SUM::runCudaVariantReduceAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    Real_ptr dsum;
    allocCudaDeviceData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initCudaDeviceData(dsum, &sum_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      reduce_sum<block_size><<<grid_size, block_size,
                  sizeof(Real_type)*block_size>>>( x,
                                                   dsum, sum_init,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

      Real_type lsum;
      Real_ptr plsum = &lsum;
      getCudaDeviceData(plsum, dsum, 1);

      m_sum = lsum;

    }
    stopTimer();

    deallocCudaDeviceData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    Real_ptr dsum;
    allocCudaDeviceData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initCudaDeviceData(dsum, &sum_init, 1);

      auto reduce_sum_lambda = [=] __device__() {
        REDUCE_SUM_BODY_CUDA(::atomicAdd)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda<block_size><<<grid_size, block_size,
                  sizeof(Real_type)*block_size>>>( reduce_sum_lambda );
      cudaErrchk( cudaGetLastError() );

      Real_type lsum;
      Real_ptr plsum = &lsum;
      getCudaDeviceData(plsum, dsum, 1);

      m_sum = lsum;

    }
    stopTimer();

    deallocCudaDeviceData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce_atomic, Real_type> sum(sum_init);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          REDUCE_SUM_BODY;
      });

      m_sum = sum.get();

    }
    stopTimer();

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void REDUCE_SUM::runCudaVariantReduce(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> sum(sum_init);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          REDUCE_SUM_BODY;
      });

      m_sum = sum.get();

    }
    stopTimer();

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;

  }

}


template < size_t block_size >
void REDUCE_SUM::runCudaVariantReduceExperimental(VariantID vid, size_t exp)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_CUDA && exp == 0) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    Real_ptr dsum;
    allocCudaReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      reduce_sum_exp0<block_size><<<grid_size, block_size>>>( x,
                                                   dsum, sum_init,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk(cudaStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocCudaReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else if ( vid == Base_CUDA && exp == 1) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    Real_ptr dsum;
    allocCudaReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      reduce_sum_exp1<block_size><<<grid_size, block_size>>>( x,
                                                   dsum, sum_init,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk(cudaStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocCudaReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else if ( vid == Base_CUDA && exp == 2) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    Real_ptr dsum;
    allocCudaReducerData(dsum, block_size / RAJAPERF_CUDA_WARP);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t si = 0; si < block_size / RAJAPERF_CUDA_WARP; ++si) {
        dsum[si] = sum_init;
      }

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      reduce_sum_exp2<block_size><<<grid_size, block_size>>>( x,
                                                   dsum, sum_init,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk(cudaStreamSynchronize(0));

      Real_type sum = sum_init;
      for (size_t si = 0; si < block_size / RAJAPERF_CUDA_WARP; ++si) {
        sum += dsum[si];
      }
      m_sum = sum;

    }
    stopTimer();

    deallocCudaReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else if ( vid == Base_CUDA && exp == 3) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    Real_ptr dsum;
    allocCudaReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      reduce_sum_exp3<block_size><<<grid_size, block_size,
                  sizeof(Real_type)*block_size>>>( x,
                                                   dsum, sum_init,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk(cudaStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocCudaReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else if ( vid == Base_CUDA && exp == 4) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    Real_ptr dsum;
    allocCudaReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      reduce_sum_exp4<block_size><<<grid_size, block_size,
                  sizeof(Real_type)*block_size>>>( x,
                                                   dsum, sum_init,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk(cudaStreamSynchronize(0));

      m_sum = dsum[0];
    }
    stopTimer();

    deallocCudaReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else if ( vid == Base_CUDA && exp == 5) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    Real_ptr dsum;
    allocCudaReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      reduce_sum_exp5<block_size><<<grid_size, block_size,
                  sizeof(Real_type)*block_size>>>( x,
                                                   dsum, sum_init,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk(cudaStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocCudaReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else if ( vid == Base_CUDA && exp == 6) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    Real_ptr dsum;
    allocCudaReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      reduce_sum_exp6<block_size><<<grid_size, block_size,
                  sizeof(Real_type)*RAJAPERF_CUDA_WARP>>>( x,
                                                   dsum, sum_init,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk(cudaStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocCudaReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else if ( vid == Base_CUDA && exp == 7) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    Real_ptr dsum;
    allocCudaReducerData(dsum, 1);

    int dev = 0;
    cudaDeviceProp deviceProp;
    int blocks_per_multiProcessor = 0;
    cudaErrchk(cudaGetDevice(&dev));
    cudaErrchk(cudaGetDeviceProperties(&deviceProp, dev));
    cudaErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiProcessor,
        (reduce_sum_exp7<block_size>), block_size, sizeof(Real_type)*RAJAPERF_CUDA_WARP));
    const size_t max_grid_size = deviceProp.multiProcessorCount*blocks_per_multiProcessor;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = std::min(RAJA_DIVIDE_CEILING_INT(iend, block_size), max_grid_size);
      reduce_sum_exp7<block_size><<<grid_size, block_size,
                  sizeof(Real_type)*RAJAPERF_CUDA_WARP>>>( x,
                                                   dsum, sum_init,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk(cudaStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocCudaReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

  } else if ( vid == Base_CUDA && exp == 8) {

    REDUCE_SUM_DATA_SETUP_CUDA;

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
    const size_t cnt_size = 1;
    const size_t tmp_size = grid_size;

    Real_ptr dsum;
    allocCudaReducerData(dsum, 1);
    unsigned hcnt[cnt_size] = {0u};
    unsigned* dcnt;
    allocAndInitCudaDeviceData(dcnt, hcnt, cnt_size);
    Real_ptr dtmp;
    allocCudaDeviceData(dtmp, tmp_size);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      reduce_sum_exp8<block_size><<<grid_size, block_size,
                  sizeof(Real_type)*RAJAPERF_CUDA_WARP>>>( x,
                                                   dsum, sum_init,
                                                   dcnt, dtmp, iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk(cudaStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocCudaDeviceData(dtmp);
    deallocCudaDeviceData(dcnt);
    deallocCudaReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_CUDA;

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

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0 ||
        run_params.validGPUBlockSize(block_size)) {

      if (tune_idx == t) {

        runCudaVariantReduceAtomic<block_size>(vid);

      }

      t += 1;

    }

  });

  if (vid == RAJA_CUDA) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0 ||
          run_params.validGPUBlockSize(block_size)) {

        if (tune_idx == t) {

          runCudaVariantReduce<block_size>(vid);

        }

        t += 1;

      }

    });

  }

  if ( vid == Base_CUDA ) {

    for (size_t exp = 0; exp < num_cuda_exp; ++exp) {

      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {

          if (tune_idx == t) {

            runCudaVariantReduceExperimental<block_size>(vid, exp);

          }

          t += 1;

        }

      });

    }

  }

}



void REDUCE_SUM::setCudaTuningDefinitions(VariantID vid)
{
  if ( vid == Base_CUDA ) {

    addVariantTuningName(vid, "cub");

  }

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0 ||
        run_params.validGPUBlockSize(block_size)) {

      addVariantTuningName(vid, "reduceAtomic_"+std::to_string(block_size));

    }

  });

  if (vid == RAJA_CUDA) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0 ||
          run_params.validGPUBlockSize(block_size)) {

        addVariantTuningName(vid, "reduce_"+std::to_string(block_size));

      }

    });

  }

  if ( vid == Base_CUDA ) {

    for (size_t exp = 0; exp < num_cuda_exp; ++exp) {

      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {

          addVariantTuningName(vid, "reduceExp"+std::to_string(exp)+"_"+std::to_string(block_size));

        }

      });

    }

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
