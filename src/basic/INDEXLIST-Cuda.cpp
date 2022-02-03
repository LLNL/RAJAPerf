//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST.hpp"

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
  const size_t warp_size = 32;


#define INDEXLIST_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, iend); \
  allocAndInitCudaDeviceData(list, m_list, iend);

#define INDEXLIST_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_list, list, iend); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(list);

struct pair
{
  Index_type first, second;
};


// perform a warp scan on inc and return the inclusive result at each thread
__device__ Index_type warp_scan_inclusive(const Index_type inc)
{
  const int warp_index = (threadIdx.x % warp_size);

  Index_type val = inc;

  // NOTE: only works for powers of 2
  for ( int i = 1; i < warp_size; i *= 2 ) {
    const bool participate = warp_index & i;
    const int prior_id = (warp_index & ~(i-1)) - 1;
    const Index_type prior_val = __shfl_sync(0xffffffffu, val, prior_id);
    if ( participate ) {
      val = prior_val + val;
    }
  }

  return val;
}

// perform a block scan on inc and return the result at each thread
// pair.first is the exclusive result and pair.second is the inclusive result
__device__ pair block_scan(const Index_type inc)
{
  extern __shared__ volatile Index_type s_thread_counts[ ];

  Index_type val = inc;
  s_thread_counts[ threadIdx.x ] = val;
  __syncthreads();

  // NOTE: only works for powers of 2
  for ( int i = 1; i < blockDim.x; i *= 2 ) {
    const bool participate = threadIdx.x & i;
    const int prior_id = (threadIdx.x & ~(i-1)) - 1;
    if ( participate ) {
      val = s_thread_counts[ prior_id ] + s_thread_counts[ threadIdx.x ];
      s_thread_counts[ threadIdx.x ] = val;
    }
    __syncthreads();
  }

  Index_type prior_val = (threadIdx.x > 0) ? s_thread_counts[threadIdx.x-1] : 0;
  __syncthreads();

  return pair { prior_val, val };
}

// perform a grid scan on inc and return the result at each thread
// pair.first is the exclusive result and pair.second is the inclusive result
__device__ pair grid_scan(const int block_id,
                          const Index_type inc,
                          Index_type* block_counts,
                          Index_type* grid_counts,
                          unsigned* block_readys)
{
  const bool first_block = (block_id == 0);
  const bool last_block = (block_id == gridDim.x-1);
  const bool last_thread = (threadIdx.x == blockDim.x-1);
  const bool last_warp = (threadIdx.x >= blockDim.x - warp_size);
  const int warp_index = (threadIdx.x % warp_size);
  const int warp_index_mask = (1u << warp_index);
  const int warp_index_mask_right = warp_index_mask | (warp_index_mask - 1);

  pair count = block_scan(inc);

  if (first_block) {

    if (!last_block && last_thread) {
      block_counts[block_id] = count.second;  // write inclusive scan result for block
      grid_counts[block_id] = count.second;   // write inclusive scan result for grid through block
      __threadfence();                          // ensure block_counts, grid_counts ready (release)
      atomicExch(&block_readys[block_id], 2u); // write block_counts, grid_counts are ready
    }

  } else {

    if (!last_block && last_thread) {
      block_counts[block_id] = count.second;  // write inclusive scan result for block
      __threadfence();                          // ensure block_counts ready (release)
      atomicExch(&block_readys[block_id], 1u); // write block_counts is ready
    }

    __shared__ volatile Index_type s_prev_grid_count;

    // get prev_grid_count using last warp in block
    if (last_warp) {

      Index_type prev_block_count = 0;

      const int prev_block_id = block_id-warp_size+warp_index;

      unsigned prev_block_ready = (prev_block_id >= 0) ? 0u : 1u;
      unsigned prev_blocks_ready_ballot = 0u;
      unsigned prev_grids_ready_ballot = 0u;

      // ensure previous block_counts are ready and at least one grid_count is ready
      do {
        if (prev_block_id >= 0 && prev_block_ready != 2u) {
          prev_block_ready = atomicCAS(&block_readys[prev_block_id], 11u, 11u);
        }

        prev_blocks_ready_ballot = __ballot_sync(0xffffffffu, prev_block_ready >= 1u);
        prev_grids_ready_ballot = __ballot_sync(0xffffffffu, prev_block_ready == 2u);

      } while (prev_blocks_ready_ballot != 0xffffffffu || prev_grids_ready_ballot == 0u);
      __threadfence(); // ensure block_counts or grid_counts ready (acquire)

      // read one grid_count from a block with id grid_count_ready_id
      // and read the block_counts from blocks with higher ids.
      if (warp_index_mask > prev_grids_ready_ballot) {
        // get block_counts for prev_block_ids in (grid_count_ready_id, block_id)
        prev_block_count = block_counts[prev_block_id];
      } else if (prev_grids_ready_ballot == (prev_grids_ready_ballot & warp_index_mask_right)) {
        // get grid_count for grid_count_ready_id
        prev_block_count = grid_counts[prev_block_id];
      }

      Index_type prev_grid_count = warp_scan_inclusive(prev_block_count);

      if (last_thread) {

        if (!last_block) {
          grid_counts[block_id] = prev_grid_count + count.second;   // write inclusive scan result for grid through block
          __threadfence();                        // ensure grid_counts ready (release)
          atomicExch(&block_readys[block_id], 2u); // write grid_counts is ready
        }

        s_prev_grid_count = prev_grid_count;
      }
    }

    __syncthreads();
    Index_type prev_grid_count = s_prev_grid_count;

    count.first  = prev_grid_count + count.first;
    count.second = prev_grid_count + count.second;

    if (last_block) {
      for (int i = threadIdx.x; i < gridDim.x-1; i += blockDim.x) {
        while (atomicCAS(&block_readys[i], 2u, 0u) != 2u);
      }
    }
  }

  return count;
}

__device__ int get_block_id(unsigned* block_id_inc)
{
  __shared__ volatile unsigned s_block_id;
  if (threadIdx.x == 0) {
    s_block_id = atomicInc(block_id_inc, gridDim.x-1);
  }
  __syncthreads();
  unsigned block_id = s_block_id;
  __syncthreads();
  return static_cast<int>(block_id);
}

__global__ void indexlist(Real_ptr x,
                          Int_ptr list,
                          Index_type* block_counts,
                          Index_type* grid_counts,
                          unsigned* block_readys,
                          unsigned* block_id_inc,
                          Index_type* len,
                          Index_type iend)
{
  const int block_id = get_block_id(block_id_inc);

  Index_type i = block_id * blockDim.x + threadIdx.x;
  Index_type inc = 0;
  if (i < iend) {
    if (INDEXLIST_CONDITIONAL) {
      inc = 1;
    }
  }

  pair count = grid_scan(block_id, inc, block_counts, grid_counts, block_readys);

  if (i < iend) {
    if (count.first != count.second) {
      list[count.first] = i;
    }
    if (i == iend-1) {
      *len = count.second;
    }
  }
}

void INDEXLIST::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INDEXLIST_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    INDEXLIST_DATA_SETUP_CUDA;

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT((iend-ibegin), block_size);

    Index_type* len;
    allocCudaPinnedData(len, 1);
    Index_type* block_counts;
    allocCudaDeviceData(block_counts, grid_size);
    Index_type* grid_counts;
    allocCudaDeviceData(grid_counts, grid_size);
    unsigned* block_readys;
    allocCudaDeviceData(block_readys, grid_size);
    cudaErrchk( cudaMemset(block_readys, 0, sizeof(unsigned)*grid_size) );
    unsigned* block_id_inc;
    allocCudaDeviceData(block_id_inc, grid_size);
    cudaErrchk( cudaMemset(block_id_inc, 0, sizeof(unsigned)) );

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      indexlist<<<grid_size, block_size, sizeof(Index_type)*block_size>>>(
          x+ibegin, list+ibegin,
          block_counts, grid_counts, block_readys, block_id_inc,
          len, iend-ibegin );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk( cudaDeviceSynchronize() );
      m_len = *len;

    }
    stopTimer();

    deallocCudaPinnedData(len);
    deallocCudaDeviceData(block_counts);
    deallocCudaDeviceData(grid_counts);
    deallocCudaDeviceData(block_readys);
    deallocCudaDeviceData(block_id_inc);

    INDEXLIST_DATA_TEARDOWN_CUDA;

  } else {
    std::cout << "\n  INDEXLIST : Unknown variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
