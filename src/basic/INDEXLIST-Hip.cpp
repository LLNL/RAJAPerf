//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <rocprim/block/block_scan.hpp>
#include <rocprim/block/block_exchange.hpp>
#include <rocprim/warp/warp_reduce.hpp>
#include <rocprim/warp/warp_scan.hpp>

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define magic numbers for HIP execution
  //
  const size_t warp_size = 64;
  const size_t items_per_thread = 8;


// perform a grid scan on val and returns the result at each thread
// in exclusive and inclusive, note that val is used as scratch space
template < size_t block_size, size_t items_per_thread >
__device__ void grid_scan(const int block_id,
                          Index_type (&val)[items_per_thread],
                          Index_type (&exclusive)[items_per_thread],
                          Index_type (&inclusive)[items_per_thread],
                          Index_type* block_counts,
                          Index_type* grid_counts,
                          unsigned* block_readys)
{
  const bool first_block = (block_id == 0);
  const bool last_block = (block_id == static_cast<int>(gridDim.x-1));
  const bool last_thread = (threadIdx.x == block_size-1);
  const bool last_warp = (threadIdx.x >= block_size - warp_size);
  const int warp_index = (threadIdx.x % warp_size);
  const unsigned long long warp_index_mask = (1ull << warp_index);
  const unsigned long long warp_index_mask_right = warp_index_mask | (warp_index_mask - 1ull);

  using BlockScan = rocprim::block_scan<Index_type, block_size>; //, rocprim::block_scan_algorithm::reduce_then_scan>;
  using BlockExchange = rocprim::block_exchange<Index_type, block_size, items_per_thread>;
  using WarpReduce = rocprim::warp_reduce<Index_type, warp_size>;

  union SharedStorage {
    typename BlockScan::storage_type block_scan_storage;
    typename BlockExchange::storage_type block_exchange_storage;
    typename WarpReduce::storage_type warp_reduce_storage;
    volatile Index_type prev_grid_count;
  };
  __shared__ SharedStorage s_temp_storage;


  BlockExchange().striped_to_blocked(val, val, s_temp_storage.block_exchange_storage);
  __syncthreads();


  BlockScan().exclusive_scan(val, exclusive, Index_type{0}, s_temp_storage.block_scan_storage);
  __syncthreads();

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    inclusive[ti] = exclusive[ti] + val[ti];
  }

  BlockExchange().blocked_to_striped(exclusive, exclusive, s_temp_storage.block_exchange_storage);
  __syncthreads();
  BlockExchange().blocked_to_striped(inclusive, inclusive, s_temp_storage.block_exchange_storage);
  __syncthreads();
  if (first_block) {

    if (!last_block && last_thread) {
      block_counts[block_id] = inclusive[items_per_thread-1]; // write inclusive scan result for block
      grid_counts[block_id] = inclusive[items_per_thread-1];  // write inclusive scan result for grid through block
      __threadfence();                         // ensure block_counts, grid_counts ready (release)
      atomicExch(&block_readys[block_id], 2u); // write block_counts, grid_counts are ready
    }

  } else {

    if (!last_block && last_thread) {
      block_counts[block_id] = inclusive[items_per_thread-1]; // write inclusive scan result for block
      __threadfence();                         // ensure block_counts ready (release)
      atomicExch(&block_readys[block_id], 1u); // write block_counts is ready
    }

    // get prev_grid_count using last warp in block
    if (last_warp) {

      Index_type prev_grid_count = 0;

      // accumulate previous block counts into registers of warp

      int prev_block_base_id = block_id - warp_size;

      unsigned prev_block_ready = 0u;
      unsigned long long prev_blocks_ready_ballot = 0ull;
      unsigned long long prev_grids_ready_ballot = 0ull;

      // accumulate full warp worths of block counts
      // stop if run out of full warps of a grid count is ready
      while (prev_block_base_id >= 0) {

        const int prev_block_id = prev_block_base_id + warp_index;

        // ensure previous block_counts are ready
        do {
          prev_block_ready = atomicCAS(&block_readys[prev_block_id], 11u, 11u);

          prev_blocks_ready_ballot = __ballot(prev_block_ready >= 1u);

        } while (prev_blocks_ready_ballot != 0xffffffffffffffffull);

        prev_grids_ready_ballot = __ballot(prev_block_ready == 2u);

        if (prev_grids_ready_ballot != 0ull) {
          break;
        }

        __threadfence(); // ensure block_counts or grid_counts ready (acquire)

        // accumulate block_counts for prev_block_id
        prev_grid_count += block_counts[prev_block_id];

        prev_block_ready = 0u;

        prev_block_base_id -= warp_size;
      }

      const int prev_block_id = prev_block_base_id + warp_index;

      // ensure previous block_counts are ready
      // this checks that block counts is ready for all blocks above
      // the highest grid count that is ready
      while (~prev_blocks_ready_ballot >= prev_grids_ready_ballot) {

        if (prev_block_id >= 0) {
          prev_block_ready = atomicCAS(&block_readys[prev_block_id], 11u, 11u);
        }

        prev_blocks_ready_ballot = __ballot(prev_block_ready >= 1u);
        prev_grids_ready_ballot = __ballot(prev_block_ready == 2u);
      }
      __threadfence(); // ensure block_counts or grid_counts ready (acquire)

      // read one grid_count from a block with id grid_count_ready_id
      // and read the block_counts from blocks with higher ids.
      if (warp_index_mask > prev_grids_ready_ballot) {
        // accumulate block_counts for prev_block_id
        prev_grid_count += block_counts[prev_block_id];
      } else if (prev_grids_ready_ballot == (prev_grids_ready_ballot & warp_index_mask_right)) {
        // accumulate grid_count for grid_count_ready_id
        prev_grid_count += grid_counts[prev_block_id];
      }


      WarpReduce().reduce(prev_grid_count, prev_grid_count, s_temp_storage.warp_reduce_storage);
      prev_grid_count = __shfl(prev_grid_count, 0, warp_size); // broadcast output to all threads in warp

      if (last_thread) {

        if (!last_block) {
          grid_counts[block_id] = prev_grid_count + inclusive[items_per_thread-1];   // write inclusive scan result for grid through block
          __threadfence();                        // ensure grid_counts ready (release)
          atomicExch(&block_readys[block_id], 2u); // write grid_counts is ready
        }

        s_temp_storage.prev_grid_count = prev_grid_count;
      }
    }

    __syncthreads();
    Index_type prev_grid_count = s_temp_storage.prev_grid_count;

    for (size_t ti = 0; ti < items_per_thread; ++ti) {
      exclusive[ti] = prev_grid_count + exclusive[ti];
      inclusive[ti] = prev_grid_count + inclusive[ti];
    }
  }
}

template < size_t block_size, size_t items_per_thread >
__launch_bounds__(block_size)
__global__ void indexlist(Real_ptr x,
                          Int_ptr list,
                          Index_type* block_counts,
                          Index_type* grid_counts,
                          unsigned* block_readys,
                          Index_type* len,
                          Index_type iend)
{
  // It looks like blocks do not start running in order in hip, so a block
  // with a higher index can't wait on a block with a lower index without
  // deadlocking (have to replace with an atomicInc)
  const int block_id = blockIdx.x;

  Index_type vals[items_per_thread];

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    Index_type val = 0;
    if (i < iend) {
      if (INDEXLIST_CONDITIONAL) {
        val = 1;
      }
    }
    vals[ti] = val;
  }

  Index_type exclusives[items_per_thread];
  Index_type inclusives[items_per_thread];
  grid_scan<block_size, items_per_thread>(
      block_id, vals, exclusives, inclusives, block_counts, grid_counts, block_readys);

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    Index_type exclusive = exclusives[ti];
    Index_type inclusive = inclusives[ti];
    if (i < iend) {
      if (exclusive != inclusive) {
        list[exclusive] = i;
      }
      if (i == iend-1) {
        *len = inclusive;
      }
    }
  }
}

template < size_t block_size >
void INDEXLIST::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  INDEXLIST_DATA_SETUP;

  if ( vid == Base_HIP ) {

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT((iend-ibegin), block_size*items_per_thread);
    const size_t shmem_size = 0;

    Index_type* len;
    allocData(DataSpace::HipPinned, len, 1);
    Index_type* block_counts;
    allocData(DataSpace::HipDevice, block_counts, grid_size);
    Index_type* grid_counts;
    allocData(DataSpace::HipDevice, grid_counts, grid_size);
    unsigned* block_readys;
    allocData(DataSpace::HipDevice, block_readys, grid_size);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk( hipMemsetAsync(block_readys, 0, sizeof(unsigned)*grid_size, res.get_stream()) );
      indexlist<block_size, items_per_thread>
          <<<grid_size, block_size, shmem_size, res.get_stream()>>>(
          x+ibegin, list+ibegin,
          block_counts, grid_counts, block_readys,
          len, iend-ibegin );
      hipErrchk( hipGetLastError() );

      hipErrchk( hipStreamSynchronize( res.get_stream() ) );
      m_len = *len;

    }
    stopTimer();

    deallocData(DataSpace::HipPinned, len);
    deallocData(DataSpace::HipDevice, block_counts);
    deallocData(DataSpace::HipDevice, grid_counts);
    deallocData(DataSpace::HipDevice, block_readys);

  } else {
    getCout() << "\n  INDEXLIST : Unknown variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(INDEXLIST, Hip)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
