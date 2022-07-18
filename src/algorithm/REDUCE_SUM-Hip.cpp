//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#if defined(__HIPCC__)
#define ROCPRIM_HIP_API 1
#include "rocprim/device/device_reduce.hpp"
#elif defined(__CUDACC__)
#include "cub/device/device_reduce.cuh"
#include "cub/util_allocator.cuh"
#endif

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

#define REDUCE_SUM_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, iend);

#define REDUCE_SUM_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(x);

#define REDUCE_SUM_BODY_HIP(atomicAdd) \
  RAJAPERF_REDUCE_1_HIP(Real_type, REDUCE_SUM_VAL, dsum, sum_init, RAJAPERF_ADD_OP, atomicAdd)

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  REDUCE_SUM_BODY_HIP(::atomicAdd)
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_unsafe(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  REDUCE_SUM_BODY_HIP(RAJAPERF_HIP_unsafeAtomicAdd)
}

constexpr size_t num_hip_exp = 15;

// block gets per thread values
// grid reduction (non-reproduceable)
//     threads atomically add into a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp0(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  Real_type val = sum_init;

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  RAJAPERF_HIP_unsafeAtomicAdd( dsum, val );
}

// block gets per thread values
// warp reduction (reproduceable)
//     warps reduce to single value using shfl instructions
// grid reduction (non-reproduceable)
//     one thread per warp atomically adds into a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp1(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  Real_type val = sum_init;

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
    val = (val + __shfl_xor(val, i));
  }

  if ( threadIdx.x % RAJAPERF_HIP_WAVEFRONT == 0u ) {
    RAJAPERF_HIP_unsafeAtomicAdd( dsum, val );
  }
}

// block gets per thread values
// warp reduction (reproduceable)
//     warps reduce to a single value using shfl instructions
// partial grid reduction (block_size/warp_size parts) (non-reproduceable)
//     one thread per warp atomically adds into a location per warp
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp2(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  Real_type val = sum_init;

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
    val = (val + __shfl_xor(val, i));
  }

  if ( threadIdx.x % RAJAPERF_HIP_WAVEFRONT == 0u ) {
    RAJAPERF_HIP_unsafeAtomicAdd( &dsum[threadIdx.x / RAJAPERF_HIP_WAVEFRONT], val );
  }
}

// block gets per thread values
// block reduction (reproduceable)
//     blocks reduce to a single value using binary reduction tree in shmem
// grid reduction (non-reproduceable)
//     one thread per block atomically adds into a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp3(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    Real_type val = sum_init;
    if ( i < iend ) {
      val = x[i];
    }

    _shmem[ threadIdx.x ] = val;
  }
  __syncthreads();

  for ( unsigned i = block_size / 2u; i > 0u; i /= 2u ) {
    if ( threadIdx.x < i ) {
      _shmem[ threadIdx.x ] = (_shmem[ threadIdx.x ] + _shmem[ threadIdx.x + i ]);
    }
     __syncthreads();
  }

  if ( threadIdx.x == 0 ) {
    RAJAPERF_HIP_unsafeAtomicAdd( dsum, _shmem[ 0 ] );
  }
}

// block gets per thread values
// block reduction (reproduceable)
//     blocks reduce to a warp-worth of values using binary reduction tree in shmem
//     one warp per block reduces to a single value using shfl instructions
// grid reduction (non-reproduceable)
//     one thread per block atomically adds into a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp4(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    Real_type val = sum_init;
    if ( i < iend ) {
      val = x[i];
    }

    _shmem[ threadIdx.x ] = val;
  }
  __syncthreads();

  for ( unsigned i = block_size / 2u; i >= RAJAPERF_HIP_WAVEFRONT; i /= 2u ) {
    if ( threadIdx.x < i ) {
      _shmem[ threadIdx.x ] = (_shmem[ threadIdx.x ] + _shmem[ threadIdx.x + i ]);
    }
     __syncthreads();
  }

  if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT) {
    Real_type val = _shmem[ threadIdx.x ];

    for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
      val = (val + __shfl_xor(val, i));
    }

    if ( threadIdx.x == 0 ) {
      RAJAPERF_HIP_unsafeAtomicAdd( dsum, val );
    }
  }
}

// block gets per thread values
// block reduction (reproduceable)
//     one warp per block reduces the block-worth of values to warp-worth of values using a warp stride loop in shmem
//     one warp per block reduces to a single value using shfl instructions
// grid reduction (non-reproduceable)
//     one thread per block atomically adds into a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp5(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    Real_type val = sum_init;
    if ( i < iend ) {
      val = x[i];
    }

    _shmem[ threadIdx.x ] = val;
  }
  __syncthreads();

  if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT) {

    Real_type val = sum_init;
    for ( unsigned i = 0; i < block_size; i += RAJAPERF_HIP_WAVEFRONT ) {
      val = (val + _shmem[ threadIdx.x + i ]);
    }

    for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
      val = (val + __shfl_xor(val, i));
    }

    if ( threadIdx.x == 0 ) {
      RAJAPERF_HIP_unsafeAtomicAdd( dsum, val );
    }
  }
}

// block gets per thread values
// block reduction (reproduceable)
//     warps in block reduce the block-worth of values to warp-worth of values using shfl instructions
//         and shmem to transfer into a single warp
//     one warp per block reduces to a single value using shfl instructions
// grid reduction (non-reproduceable)
//     one thread per block atomically adds into a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp6(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  Real_type val = sum_init;
  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  constexpr unsigned wavefront_shrink_factor = block_size / RAJAPERF_HIP_WAVEFRONT;
  constexpr unsigned wavefront_shrink_size = RAJAPERF_HIP_WAVEFRONT / wavefront_shrink_factor;
  const unsigned wavefront_idx = threadIdx.x / RAJAPERF_HIP_WAVEFRONT;
  const unsigned wavefront_thread_idx = threadIdx.x % RAJAPERF_HIP_WAVEFRONT;

  for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i >= wavefront_shrink_size; i /= 2u ) {
    val = (val + __shfl_xor(val, i));
  }

  if (wavefront_thread_idx < wavefront_shrink_size) {
    _shmem[ wavefront_thread_idx + wavefront_idx * wavefront_shrink_size ] = val;
  }

  __syncthreads();

  if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT) {
    Real_type val = _shmem[ threadIdx.x ];

    for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
      val = (val + __shfl_xor(val, i));
    }

    if ( threadIdx.x == 0 ) {
      RAJAPERF_HIP_unsafeAtomicAdd( dsum, val );
    }
  }
}

// block gets per thread values using a grid-stride loop to collect multiple values per thread
// block reduction (reproduceable)
//     warps in block reduce the block-worth of values to warp-worth of values using shfl instructions
//         and shmem to transfer into a single warp
//     one warp per block reduces to a single value using shfl instructions
// grid reduction (non-reproduceable)
//     one thread per block atomically adds into a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp7(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  Real_type val = sum_init;
  for ( Index_type i = blockIdx.x * block_size + threadIdx.x;
        i < iend ; i += gridDim.x * block_size ) {
    val = (val + x[i]);
  }

  constexpr unsigned wavefront_shrink_factor = block_size / RAJAPERF_HIP_WAVEFRONT;
  constexpr unsigned wavefront_shrink_size = RAJAPERF_HIP_WAVEFRONT / wavefront_shrink_factor;
  const unsigned wavefront_idx = threadIdx.x / RAJAPERF_HIP_WAVEFRONT;
  const unsigned wavefront_thread_idx = threadIdx.x % RAJAPERF_HIP_WAVEFRONT;

  for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i >= wavefront_shrink_size; i /= 2u ) {
    val = (val + __shfl_xor(val, i));
  }

  if (wavefront_thread_idx < wavefront_shrink_size) {
    _shmem[ wavefront_thread_idx + wavefront_idx * wavefront_shrink_size ] = val;
  }

  __syncthreads();

  if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT) {
    Real_type val = _shmem[ threadIdx.x ];

    for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
      val = (val + __shfl_xor(val, i));
    }

    if ( threadIdx.x == 0 ) {
      RAJAPERF_HIP_unsafeAtomicAdd( dsum, val );
    }
  }
}

// block gets per thread values
// block reduction (reproduceable)
//     warps in block reduce the block-worth of values to warp-worth of values using shfl instructions
//         and shmem to transfer into a single warp
//     one warp per block reduces to a single value using shfl instructions
//     one thread per block writes to "block array"
// grid reduction (reproduceable)
//     last block reduces the "block array" using above block reduction method
//     one thread writes to a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp8(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                                unsigned* dcnt, Real_ptr dtmp, Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  Real_type val = sum_init;
  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  constexpr unsigned wavefront_shrink_factor = block_size / RAJAPERF_HIP_WAVEFRONT;
  constexpr unsigned wavefront_shrink_size = RAJAPERF_HIP_WAVEFRONT / wavefront_shrink_factor;
  const unsigned wavefront_idx = threadIdx.x / RAJAPERF_HIP_WAVEFRONT;
  const unsigned wavefront_thread_idx = threadIdx.x % RAJAPERF_HIP_WAVEFRONT;

  for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i >= wavefront_shrink_size; i /= 2u ) {
    val = (val + __shfl_xor(val, i));
  }

  if (wavefront_thread_idx < wavefront_shrink_size) {
    _shmem[ wavefront_thread_idx + wavefront_idx * wavefront_shrink_size ] = val;
  }

  __syncthreads();

  __shared__ unsigned reduction_block_idx;
  if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT) {
    Real_type val = _shmem[ threadIdx.x ];

    for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
      val = (val + __shfl_xor(val, i));
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

    for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i >= wavefront_shrink_size; i /= 2u ) {
      val = (val + __shfl_xor(val, i));
    }

    if (wavefront_thread_idx < wavefront_shrink_size) {
      _shmem[ wavefront_thread_idx + wavefront_idx * wavefront_shrink_size ] = val;
    }

    __syncthreads();

    if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT) {
      Real_type val = _shmem[ threadIdx.x ];

      for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
        val = (val + __shfl_xor(val, i));
      }

      if ( threadIdx.x == 0 ) {
        dsum[0] = val;
      }
    }

  }

}

// block gets per thread values
// block reduction (reproduceable)
//     warps in block reduce the block-worth of values to warp-worth of values using shfl instructions
//         and then shmem to transfer into a single warp
//     one warp per block reduces to a single value using shfl instructions
//     one thread per block writes to "block grid array"
// sub-grid reduction (grid_size / block_size parts) (reproduceable)
//     last block in every block_size number of blocks reduces the "block grid array" using above block reduction method
//     one thread writes to "block array"
// grid reduction (reproduceable)
//     last block reduces the "block array" using above block reduction method
//     one thread writes to a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp9(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                                unsigned* dcnt_grid, Real_ptr dtmp_grid,
                                unsigned* dcnt_sub_grid, Real_ptr dtmp_sub_grid,
                                unsigned num_sub_grids, unsigned last_sub_grid_size,
                                Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  Real_type val = sum_init;
  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  constexpr unsigned wavefront_shrink_factor = block_size / RAJAPERF_HIP_WAVEFRONT;
  constexpr unsigned wavefront_shrink_size = RAJAPERF_HIP_WAVEFRONT / wavefront_shrink_factor;
  const unsigned wavefront_idx = threadIdx.x / RAJAPERF_HIP_WAVEFRONT;
  const unsigned wavefront_thread_idx = threadIdx.x % RAJAPERF_HIP_WAVEFRONT;

  const unsigned grid_idx = blockIdx.x;
  const unsigned grid_size = gridDim.x;
  const unsigned sub_grid_idx = grid_idx / block_size;
  constexpr unsigned full_sub_grid_size = block_size;
  const unsigned sub_grid_size = ((grid_idx + last_sub_grid_size) < grid_size)
                          ? block_size : last_sub_grid_size;

  for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i >= wavefront_shrink_size; i /= 2u ) {
    val = (val + __shfl_xor(val, i));
  }

  if (wavefront_thread_idx < wavefront_shrink_size) {
    _shmem[ wavefront_thread_idx + wavefront_idx * wavefront_shrink_size ] = val;
  }

  __syncthreads();

  __shared__ unsigned reduction_idx;
  if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT) {
    val = _shmem[ threadIdx.x ];

    for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
      val = (val + __shfl_xor(val, i));
    }

    if ( threadIdx.x == 0 ) {
      dtmp_grid[grid_idx] = val;
      __threadfence();
      reduction_idx = ::atomicInc(&dcnt_grid[sub_grid_idx], sub_grid_size-1);
      __threadfence();
    }
  }

  __syncthreads();

  if (reduction_idx == sub_grid_size-1) {

    val = sum_init;
    if (threadIdx.x < sub_grid_size) {
      val = dtmp_grid[threadIdx.x + sub_grid_idx * full_sub_grid_size];
    }

    for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i >= wavefront_shrink_size; i /= 2u ) {
      val = (val + __shfl_xor(val, i));
    }

    if (wavefront_thread_idx < wavefront_shrink_size) {
      _shmem[ wavefront_thread_idx + wavefront_idx * wavefront_shrink_size ] = val;
    }

    __syncthreads();

    __shared__ unsigned reduction_idx;
    if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT) {
      val = _shmem[ threadIdx.x ];

      for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
        val = (val + __shfl_xor(val, i));
      }

      if ( threadIdx.x == 0 ) {
        dtmp_sub_grid[sub_grid_idx] = val;
        __threadfence();
        reduction_idx = ::atomicInc(dcnt_sub_grid, num_sub_grids-1);
        __threadfence();
      }
    }

    __syncthreads();

    if (reduction_idx == num_sub_grids-1) {

      val = sum_init;
      for ( unsigned i = threadIdx.x ; i < num_sub_grids; i += block_size ) {
        val = (val + dtmp_sub_grid[i]);
      }

      for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i >= wavefront_shrink_size; i /= 2u ) {
        val = (val + __shfl_xor(val, i));
      }

      if (wavefront_thread_idx < wavefront_shrink_size) {
        _shmem[ wavefront_thread_idx + wavefront_idx * wavefront_shrink_size ] = val;
      }

      __syncthreads();

      if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT) {
        val = _shmem[ threadIdx.x ];

        for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
          val = (val + __shfl_xor(val, i));
        }

        if ( threadIdx.x == 0 ) {
          dsum[0] = val;
        }
      }

    }
  }

}

// block gets per thread values
// warp reduction (non-reproduceable)
//     warps reduce their warp-worth of values to a single value using atomics in shmem
// grid reduction (non-reproduceable)
//     one thread per warp atomically adds into a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp10(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  Real_type val = sum_init;
  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  const unsigned wavefront_idx = threadIdx.x / RAJAPERF_HIP_WAVEFRONT;
  const unsigned wavefront_thread_idx = threadIdx.x % RAJAPERF_HIP_WAVEFRONT;

  if ( wavefront_thread_idx == 0u ) {
    _shmem[wavefront_idx] = sum_init;
  }

  RAJAPERF_HIP_unsafeAtomicAdd(&_shmem[wavefront_idx], val);

  if ( wavefront_thread_idx == 0u ) {
    RAJAPERF_HIP_unsafeAtomicAdd( dsum, _shmem[wavefront_idx] );
  }
}

// block gets per thread values
// block reduction (non-reproduceable)
//     blocks reduce their block-worth of values to a single value using atomics in shmem
// grid reduction (non-reproduceable)
//     one thread per block atomically adds into a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp11(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  Real_type val = sum_init;
  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  if ( threadIdx.x == 0u ) {
    _shmem[0] = sum_init;
  }

  __syncthreads();

  RAJAPERF_HIP_unsafeAtomicAdd(&_shmem[0], val);

  __syncthreads();

  if ( threadIdx.x == 0u ) {
    RAJAPERF_HIP_unsafeAtomicAdd( dsum, _shmem[0] );
  }
}

// block gets per thread values
// block reduction (non-reproduceable)
//     warps reduce their warp-worth of values to a single value using shfl instructions
//     warps reduce their single values to a single value using atomics in shmem
// grid reduction (non-reproduceable)
//     one thread per block atomically adds into a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp12(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  Real_type val = sum_init;
  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
    val = (val + __shfl_xor(val, i));
  }

  const unsigned wavefront_thread_idx = threadIdx.x % RAJAPERF_HIP_WAVEFRONT;

  if ( threadIdx.x == 0u ) {
    _shmem[0] = sum_init;
  }

  __syncthreads();

  if (wavefront_thread_idx == 0u) {
    RAJAPERF_HIP_unsafeAtomicAdd(&_shmem[0], val);
  }

  __syncthreads();

  if ( threadIdx.x == 0u ) {
    RAJAPERF_HIP_unsafeAtomicAdd( dsum, _shmem[0] );
  }
}

// block gets per thread values
// block reduction (non-reproduceable)
//     warps reduce their warp-worth of values to a warp-worth of values using atomics in shmem
//     one warp reduces its values to a single value using shfl instructions
// grid reduction (non-reproduceable)
//     one thread per block atomically adds into a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp13(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  Real_type val = sum_init;
  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  const unsigned wavefront_thread_idx = threadIdx.x % RAJAPERF_HIP_WAVEFRONT;

  if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT ) {
    _shmem[threadIdx.x] = sum_init;
  }

  __syncthreads();

  RAJAPERF_HIP_unsafeAtomicAdd(&_shmem[wavefront_thread_idx], val);

  __syncthreads();

  if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT ) {

    val = _shmem[threadIdx.x];

    for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
      val = (val + __shfl_xor(val, i));
    }

    if ( threadIdx.x == 0u ) {
      RAJAPERF_HIP_unsafeAtomicAdd( dsum, val );
    }

  }
}

// block gets per thread values
// block reduction (non-reproduceable)
//     warps reduce their warp-worth of values to a warp-worth of values using atomics in shmem
//     one warp reduces its values to a single value using shfl instructions
// grid reduction (non-reproduceable)
//     one thread per block atomically adds into a single location
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum_exp14(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, _shmem);

  Real_type val = sum_init;
  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;

    if ( i < iend ) {
      val = x[i];
    }
  }

  if (block_size > RAJAPERF_HIP_WAVEFRONT) {

    const unsigned wavefront_thread_idx = threadIdx.x % RAJAPERF_HIP_WAVEFRONT;

    if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT ) {
      _shmem[threadIdx.x] = val;
    }

    __syncthreads();

    if ( !(threadIdx.x < RAJAPERF_HIP_WAVEFRONT) ) {
      RAJAPERF_HIP_unsafeAtomicAdd(&_shmem[wavefront_thread_idx], val);
    }

    __syncthreads();

    if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT ) {

      val = _shmem[threadIdx.x];

    }

  }

  if ( threadIdx.x < RAJAPERF_HIP_WAVEFRONT ) {

    for ( unsigned i = RAJAPERF_HIP_WAVEFRONT / 2u; i > 0u; i /= 2u ) {
      val = (val + __shfl_xor(val, i));
    }

    if ( threadIdx.x == 0u ) {
      RAJAPERF_HIP_unsafeAtomicAdd( dsum, val );
    }

  }
}


void REDUCE_SUM::runHipVariantRocprim(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    REDUCE_SUM_DATA_SETUP_HIP;

    hipStream_t stream = 0;

    int len = iend - ibegin;

    Real_type* sum_storage;
    allocHipReducerData(sum_storage, 1);

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
    hipErrchk(::rocprim::reduce(d_temp_storage,
                                temp_storage_bytes,
                                x+ibegin,
                                sum_storage,
                                sum_init,
                                len,
                                rocprim::plus<Real_type>(),
                                stream));
#elif defined(__CUDACC__)
    hipErrchk(::cub::DeviceReduce::Reduce(d_temp_storage,
                                          temp_storage_bytes,
                                          x+ibegin,
                                          sum_storage,
                                          len,
                                          ::cub::Sum(),
                                          sum_init,
                                          stream));
#endif

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocHipDeviceData(temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Run
#if defined(__HIPCC__)
      hipErrchk(::rocprim::reduce(d_temp_storage,
                                  temp_storage_bytes,
                                  x+ibegin,
                                  sum_storage,
                                  sum_init,
                                  len,
                                  rocprim::plus<Real_type>(),
                                  stream));
#elif defined(__CUDACC__)
      hipErrchk(::cub::DeviceReduce::Reduce(d_temp_storage,
                                            temp_storage_bytes,
                                            x+ibegin,
                                            sum_storage,
                                            len,
                                            ::cub::Sum(),
                                            sum_init,
                                            stream));
#endif

      hipErrchk(hipStreamSynchronize(stream));
      m_sum = *sum_storage;

    }
    stopTimer();

    // Free temporary storage
    deallocHipDeviceData(temp_storage);
    deallocHipReducerData(sum_storage);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Hip variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void REDUCE_SUM::runHipVariantReduceAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    REDUCE_SUM_DATA_SETUP_HIP;

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (reduce_sum<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*block_size, 0,
                          x, dsum, sum_init, iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    REDUCE_SUM_DATA_SETUP_HIP;

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      auto reduce_sum_lambda = [=] __device__ () {
        REDUCE_SUM_BODY_HIP(::atomicAdd)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (lambda_hip<block_size, decltype(reduce_sum_lambda)>),
                          dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*block_size, 0,
                          reduce_sum_lambda );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    REDUCE_SUM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce_atomic, Real_type> sum(sum_init);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          REDUCE_SUM_BODY;
      });

      m_sum = sum.get();

    }
    stopTimer();

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Hip variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void REDUCE_SUM::runHipVariantReduceUnsafeAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    REDUCE_SUM_DATA_SETUP_HIP;

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (reduce_sum_unsafe<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*block_size, 0,
                          x, dsum, sum_init, iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    REDUCE_SUM_DATA_SETUP_HIP;

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      auto reduce_sum_lambda = [=] __device__ () {
        REDUCE_SUM_BODY_HIP(RAJAPERF_HIP_unsafeAtomicAdd)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (lambda_hip<block_size, decltype(reduce_sum_lambda)>),
                          dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*block_size, 0,
                          reduce_sum_lambda );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Hip variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void REDUCE_SUM::runHipVariantReduce(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    REDUCE_SUM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce, Real_type> sum(sum_init);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          REDUCE_SUM_BODY;
      });

      m_sum = sum.get();

    }
    stopTimer();

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Hip variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void REDUCE_SUM::runHipVariantReduceExperimental(VariantID vid, size_t exp)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_HIP && exp == 0) {

    REDUCE_SUM_DATA_SETUP_HIP;

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (reduce_sum_exp0<block_size>), dim3(grid_size), dim3(block_size),
                          0, 0,
                          x, dsum, sum_init, iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 1) {

    REDUCE_SUM_DATA_SETUP_HIP;

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (reduce_sum_exp1<block_size>), dim3(grid_size), dim3(block_size),
                          0, 0,
                          x, dsum, sum_init, iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 2) {

    REDUCE_SUM_DATA_SETUP_HIP;

    Real_ptr dsum;
    allocHipReducerData(dsum, block_size / RAJAPERF_HIP_WAVEFRONT);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t si = 0; si < block_size / RAJAPERF_HIP_WAVEFRONT; ++si) {
        dsum[si] = sum_init;
      }

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (reduce_sum_exp2<block_size>), dim3(grid_size), dim3(block_size),
                          0, 0,
                          x, dsum, sum_init, iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      Real_type sum = sum_init;
      for (size_t si = 0; si < block_size / RAJAPERF_HIP_WAVEFRONT; ++si) {
        sum += dsum[si];
      }
      m_sum = sum;

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 3) {

    REDUCE_SUM_DATA_SETUP_HIP;

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (reduce_sum_exp3<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*block_size, 0,
                          x, dsum, sum_init, iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 4) {

    REDUCE_SUM_DATA_SETUP_HIP;

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (reduce_sum_exp4<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*block_size, 0,
                          x, dsum, sum_init, iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 5) {

    REDUCE_SUM_DATA_SETUP_HIP;

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (reduce_sum_exp5<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*block_size, 0,
                          x, dsum, sum_init, iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 6) {

    REDUCE_SUM_DATA_SETUP_HIP;

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (reduce_sum_exp6<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*RAJAPERF_HIP_WAVEFRONT, 0,
                          x, dsum, sum_init, iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 7 ) {

    REDUCE_SUM_DATA_SETUP_HIP;

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    int dev = 0;
    hipDeviceProp_t deviceProp;
    int blocks_per_multiProcessor = 0;
    hipErrchk(hipGetDevice(&dev));
    hipErrchk(hipGetDeviceProperties(&deviceProp, dev));
    hipErrchk(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiProcessor,
        (reduce_sum_exp7<block_size>), block_size, sizeof(Real_type)*RAJAPERF_HIP_WAVEFRONT));
    const size_t max_grid_size = deviceProp.multiProcessorCount*blocks_per_multiProcessor;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      const size_t grid_size = std::min(RAJA_DIVIDE_CEILING_INT(iend, block_size), max_grid_size);
      hipLaunchKernelGGL( (reduce_sum_exp7<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*RAJAPERF_HIP_WAVEFRONT, 0,
                          x, dsum, sum_init, iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 8) {

    REDUCE_SUM_DATA_SETUP_HIP;

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
    const size_t cnt_size = 1;
    const size_t tmp_size = grid_size;

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);
    unsigned hcnt[cnt_size] = {0u};
    unsigned* dcnt;
    allocAndInitHipDeviceData(dcnt, hcnt, cnt_size);
    Real_ptr dtmp;
    allocHipDeviceData(dtmp, tmp_size);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      hipLaunchKernelGGL( (reduce_sum_exp8<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*RAJAPERF_HIP_WAVEFRONT, 0,
                          x, dsum, sum_init, dcnt, dtmp, iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipDeviceData(dtmp);
    deallocHipDeviceData(dcnt);
    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 9) {

    REDUCE_SUM_DATA_SETUP_HIP;

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
    constexpr size_t full_sub_grid_size = block_size;
    const size_t num_sub_grids = RAJA_DIVIDE_CEILING_INT(grid_size, full_sub_grid_size);
    const size_t last_sub_grid_size = grid_size - (num_sub_grids - 1) * full_sub_grid_size;

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);
    unsigned* dcnt_grid;
    allocHipDeviceData(dcnt_grid, num_sub_grids);
    memsetHipDeviceData(dcnt_grid, 0, num_sub_grids);
    Real_ptr dtmp_grid;
    allocHipDeviceData(dtmp_grid, grid_size);
    unsigned* dcnt_sub_grid;
    allocHipDeviceData(dcnt_sub_grid, 1);
    memsetHipDeviceData(dcnt_sub_grid, 0, 1);
    Real_ptr dtmp_sub_grid;
    allocHipDeviceData(dtmp_sub_grid, num_sub_grids);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      hipLaunchKernelGGL( (reduce_sum_exp9<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*RAJAPERF_HIP_WAVEFRONT, 0,
                          x, dsum, sum_init,
                          dcnt_grid, dtmp_grid,
                          dcnt_sub_grid, dtmp_sub_grid,
                          num_sub_grids, last_sub_grid_size,
                          iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipDeviceData(dtmp_sub_grid);
    deallocHipDeviceData(dcnt_sub_grid);
    deallocHipDeviceData(dtmp_grid);
    deallocHipDeviceData(dcnt_grid);
    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 10) {

    REDUCE_SUM_DATA_SETUP_HIP;

    const size_t num_warps = RAJA_DIVIDE_CEILING_INT(block_size, RAJAPERF_HIP_WAVEFRONT);
    const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      hipLaunchKernelGGL( (reduce_sum_exp10<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*num_warps, 0,
                          x, dsum, sum_init,
                          iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 11) {

    REDUCE_SUM_DATA_SETUP_HIP;

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      hipLaunchKernelGGL( (reduce_sum_exp11<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type), 0,
                          x, dsum, sum_init,
                          iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 12) {

    REDUCE_SUM_DATA_SETUP_HIP;

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      hipLaunchKernelGGL( (reduce_sum_exp12<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type), 0,
                          x, dsum, sum_init,
                          iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 13) {

    REDUCE_SUM_DATA_SETUP_HIP;

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      hipLaunchKernelGGL( (reduce_sum_exp13<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*RAJAPERF_HIP_WAVEFRONT, 0,
                          x, dsum, sum_init,
                          iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && exp == 14) {

    REDUCE_SUM_DATA_SETUP_HIP;

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

    Real_ptr dsum;
    allocHipReducerData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dsum[0] = sum_init;

      hipLaunchKernelGGL( (reduce_sum_exp14<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*RAJAPERF_HIP_WAVEFRONT, 0,
                          x, dsum, sum_init,
                          iend );
      hipErrchk( hipGetLastError() );

      hipErrchk(hipStreamSynchronize(0));

      m_sum = dsum[0];

    }
    stopTimer();

    deallocHipReducerData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Hip variant id = " << vid << ", exp = " << exp << std::endl;

  }

}

void REDUCE_SUM::runHipVariant(VariantID vid, size_t tune_idx)
{
  bool have_unsafe_atomics = haveHipUnsafeAtomics();

  size_t t = 0;

  if ( vid == Base_HIP ) {

    if (tune_idx == t) {

      runHipVariantRocprim(vid);

    }

    t += 1;

  }

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      if (tune_idx == t) {

        runHipVariantReduceAtomic<block_size>(vid);

      }

      t += 1;

    }

  });

  if ( vid == Base_HIP || vid == Lambda_HIP ) {

    if (have_unsafe_atomics) {

      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {

          if (tune_idx == t) {

            runHipVariantReduceUnsafeAtomic<block_size>(vid);

          }

          t += 1;

        }

      });

    }

  } else if ( vid == RAJA_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (tune_idx == t) {

          runHipVariantReduce<block_size>(vid);

        }

        t += 1;

      }

    });

  }

  if ( vid == Base_HIP ) {

    if (have_unsafe_atomics) {

      for (size_t exp = 0; exp < num_hip_exp; ++exp) {

        seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

          if (run_params.numValidGPUBlockSize() == 0u ||
              run_params.validGPUBlockSize(block_size)) {

            if (tune_idx == t) {

              runHipVariantReduceExperimental<block_size>(vid, exp);

            }

            t += 1;

          }

        });

      }

    }

  }

}

void REDUCE_SUM::setHipTuningDefinitions(VariantID vid)
{
  bool have_unsafe_atomics = haveHipUnsafeAtomics();

  if ( vid == Base_HIP ) {

#if defined(__HIPCC__)
    addVariantTuningName(vid, "rocprim");
#elif defined(__CUDACC__)
    addVariantTuningName(vid, "cub");
#endif

  }

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      addVariantTuningName(vid, "reduceAtomic_"+std::to_string(block_size));

    }

  });

  if ( vid == Base_HIP || vid == Lambda_HIP ) {

    if (have_unsafe_atomics) {

      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {

          addVariantTuningName(vid, "reduceUnsafeAtomic_"+std::to_string(block_size));

        }

      });

    }

  } else if ( vid == RAJA_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        addVariantTuningName(vid, "reduce_"+std::to_string(block_size));

      }

    });

  }

  if ( vid == Base_HIP ) {

    if (have_unsafe_atomics) {

      for (size_t exp = 0; exp < num_hip_exp; ++exp) {

        seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

          if (run_params.numValidGPUBlockSize() == 0u ||
              run_params.validGPUBlockSize(block_size)) {

            addVariantTuningName(vid, "reduceExp"+std::to_string(exp)+"_"+std::to_string(block_size));

          }

        });

      }

    }

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
