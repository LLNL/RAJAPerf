//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if defined(RAJA_ENABLE_HIP)

#include <rocprim/block/block_scan.hpp>
#include <rocprim/block/block_exchange.hpp>
#include <rocprim/warp/warp_reduce.hpp>
#include <rocprim/warp/warp_scan.hpp>

namespace rajaperf
{
namespace detail
{
namespace hip
{

//
// Define magic numbers for HIP execution
//
const size_t warp_size = 64;
const size_t max_static_shmem = 65536;
const size_t cache_line_size = (RAJA_PERFSUITE_TUNING_HIP_ARCH == 910) ? 128 : 64;


// perform a grid scan on val and returns the result at each thread
// in exclusive and inclusive, note that val is used as scratch space
template < typename DataType, size_t block_size, size_t items_per_thread >
struct GridScan
{
  using BlockScan = rocprim::block_scan<DataType, block_size>; //, rocprim::block_scan_algorithm::reduce_then_scan>;
  using BlockExchange = rocprim::block_exchange<DataType, block_size, items_per_thread>;
  using WarpReduce = rocprim::warp_reduce<DataType, warp_size>;

  using AtomicPacker = AtomicDataPacker<DataType, char>;
  using AtomicPackedType = typename AtomicPacker::type;
  static constexpr bool atomic_packable = AtomicPacker::packable;

  using DeviceStorage = std::conditional_t<atomic_packable,
      AtomicDeviceStoragePackable<AtomicPackedType, cache_line_size>,
      AtomicDeviceStorageUnpackable<DataType>>;

  union SharedStorage {
    typename BlockScan::storage_type block_scan_storage;
    typename BlockExchange::storage_type block_exchange_storage;
    typename WarpReduce::storage_type warp_reduce_storage;
    volatile DataType prev_grid_count;
  };

  static constexpr size_t shmem_size = sizeof(SharedStorage);

  // NOTE: val is clobbered by this operation
  __device__
  static void grid_scan(const int block_id,
                        DataType (&val)[items_per_thread],
                        DataType (&exclusive)[items_per_thread],
                        DataType (&inclusive)[items_per_thread],
                        DeviceStorage device_storage)
  {
    const bool first_block = (block_id == 0);
    const bool last_block = (block_id == static_cast<int>(gridDim.x-1));
    const bool last_thread = (threadIdx.x == block_size-1);
    const bool last_warp = (threadIdx.x >= block_size - warp_size);
    const int warp_index = (threadIdx.x % warp_size);
    const unsigned long long warp_index_mask = (1ull << warp_index);
    const unsigned long long warp_index_mask_right = warp_index_mask | (warp_index_mask - 1ull);

    __shared__ SharedStorage s_temp_storage;


    BlockExchange().striped_to_blocked(val, val, s_temp_storage.block_exchange_storage);
    __syncthreads();


    BlockScan().exclusive_scan(val, exclusive, DataType{0}, s_temp_storage.block_scan_storage);
    __syncthreads();

    for (size_t ti = 0; ti < items_per_thread; ++ti) {
      inclusive[ti] = exclusive[ti] + val[ti];
    }

    if (first_block) {

      if (!last_block && last_thread) {
        DataType grid_count = inclusive[items_per_thread-1];
        if constexpr(atomic_packable) {
          atomicExch(&device_storage.count_readys[block_id],
                     AtomicPacker::pack(grid_count, 2)); // write grid count with grid ready (relaxed)
        } else {
          device_storage.grid_counts[block_id] = grid_count;  // write inclusive scan result for grid through block
          __threadfence();                         // ensure block_counts, grid_counts ready (release)
          atomicExch(&device_storage.block_readys[block_id], 2u); // write block_counts, grid_counts are ready
        }
      }

    } else {

      if (!last_block && last_thread) {
        DataType block_count = inclusive[items_per_thread-1];
        if constexpr(atomic_packable) {
          atomicExch(&device_storage.count_readys[block_id],
                     AtomicPacker::pack(block_count, 1)); // write block count with block ready (relaxed)
        } else {
          device_storage.block_counts[block_id] = block_count; // write inclusive scan result for block
          __threadfence();                         // ensure block_counts ready (release)
          atomicExch(&device_storage.block_readys[block_id], 1u); // write block_counts is ready
        }
      }

    }

    BlockExchange().blocked_to_striped(exclusive, exclusive, s_temp_storage.block_exchange_storage);
    __syncthreads();
    BlockExchange().blocked_to_striped(inclusive, inclusive, s_temp_storage.block_exchange_storage);
    __syncthreads();

    if (!first_block) {

      // get prev_grid_count using last warp in block
      if (last_warp) {

        DataType prev_grid_count = 0;

        // accumulate previous block counts into registers of warp

        int prev_block_base_id = block_id - warp_size;

        DataType prev_count;
        char prev_block_ready = 0;
        unsigned long long prev_blocks_ready_ballot = 0ull;
        unsigned long long prev_grids_ready_ballot = 0ull;

        // accumulate full warp worth of block counts
        // stop if run out of full warps or a grid count is ready
        while (prev_block_base_id >= 0) {

          const int prev_block_id = prev_block_base_id + warp_index;

          // ensure previous block_counts are ready
          do {
            if constexpr(atomic_packable) {
              AtomicPackedType pack = atomicCAS(&device_storage.count_readys[prev_block_id], 11u, 11u);
              AtomicPacker::unpack(pack, prev_count, prev_block_ready);
            } else {
              prev_block_ready = atomicCAS(&device_storage.block_readys[prev_block_id], 11u, 11u);
            }

            prev_blocks_ready_ballot = __ballot(prev_block_ready >= 1);

          } while (prev_blocks_ready_ballot != 0xffffffffffffffffull);

          prev_grids_ready_ballot = __ballot(prev_block_ready == 2);

          if (prev_grids_ready_ballot != 0ull) {
            break;
          }

          // accumulate block_counts for prev_block_id
          if constexpr(!atomic_packable) {
            __threadfence(); // ensure block_counts or grid_counts ready (acquire)
            prev_count = device_storage.block_counts[prev_block_id];
          }
          prev_grid_count += prev_count;

          prev_block_ready = 0;

          prev_block_base_id -= warp_size;
        }

        const int prev_block_id = prev_block_base_id + warp_index;

        // ensure previous block_counts are ready
        // this checks that block counts is ready for all blocks above
        // the highest grid count that is ready
        while (~prev_blocks_ready_ballot >= prev_grids_ready_ballot) {

          if (prev_block_id >= 0) {
            if constexpr(atomic_packable) {
              AtomicPackedType pack = atomicCAS(&device_storage.count_readys[prev_block_id], 11u, 11u);
              AtomicPacker::unpack(pack, prev_count, prev_block_ready);
            } else {
              prev_block_ready = atomicCAS(&device_storage.block_readys[prev_block_id], 11u, 11u);
            }
          }

          prev_blocks_ready_ballot = __ballot(prev_block_ready >= 1);
          prev_grids_ready_ballot = __ballot(prev_block_ready == 2);
        }

        // read one grid_count from a block with id grid_count_ready_id
        // and read the block_counts from blocks with higher ids.
        if constexpr(atomic_packable) {
          if (warp_index_mask > prev_grids_ready_ballot) {
            // already read block_counts for prev_block_id
          } else if (prev_grids_ready_ballot == (prev_grids_ready_ballot & warp_index_mask_right)) {
            // already read grid_count for grid_count_ready_id
          } else {
            // no contribution for blocks before grid_count_ready_id
            prev_count = 0;
          }
        } else {
          __threadfence(); // ensure block_counts or grid_counts ready (acquire)
          if (warp_index_mask > prev_grids_ready_ballot) {
            // read block_counts for prev_block_id
            prev_count = device_storage.block_counts[prev_block_id];
          } else if (prev_grids_ready_ballot == (prev_grids_ready_ballot & warp_index_mask_right)) {
            // read grid_count for grid_count_ready_id
            prev_count = device_storage.grid_counts[prev_block_id];
          } else {
            // no contribution for blocks before grid_count_ready_id
            prev_count = 0;
          }
        }
        // accumulate block_counts and exactly 1 grid_count for prev_block_id
        prev_grid_count += prev_count;

        WarpReduce().reduce(prev_grid_count, prev_grid_count, s_temp_storage.warp_reduce_storage);
        prev_grid_count = __shfl(prev_grid_count, 0, warp_size); // broadcast output to all threads in warp

        if (last_thread) {

          if (!last_block) {
            DataType grid_count = prev_grid_count + inclusive[items_per_thread-1];
            if constexpr(atomic_packable) {
              atomicExch(&device_storage.count_readys[block_id],
                     AtomicPacker::pack(grid_count, 2)); // write grid count with grid ready (relaxed)
            } else {
              device_storage.grid_counts[block_id] = grid_count;   // write inclusive scan result for grid through block
              __threadfence();                        // ensure grid_counts ready (release)
              atomicExch(&device_storage.block_readys[block_id], 2u); // write grid_counts is ready
            }
          }

          s_temp_storage.prev_grid_count = prev_grid_count;
        }
      }

      __syncthreads();
      DataType prev_grid_count = s_temp_storage.prev_grid_count;

      for (size_t ti = 0; ti < items_per_thread; ++ti) {
        exclusive[ti] = prev_grid_count + exclusive[ti];
        inclusive[ti] = prev_grid_count + inclusive[ti];
      }
    }
  }

};


namespace detail
{

template < typename T, size_t block_size, size_t max_items_per_thread >
struct grid_scan_max_items_per_thread
  : std::conditional_t< (GridScan<T, block_size, max_items_per_thread>::shmem_size <= max_static_shmem),
        grid_scan_max_items_per_thread<T, block_size, max_items_per_thread+1>,
        std::integral_constant<size_t, max_items_per_thread-1> >
{
};

}

template < typename T, size_t block_size >
struct grid_scan_max_items_per_thread
  : detail::grid_scan_max_items_per_thread<T, block_size, 1>
{
};


// tune grid scan to maximize throughput while minimizing items_per_thread

// default tuning for unknown DataType or hip_arch
template < typename DataType, size_t block_size, size_t hip_arch, typename enable = void >
struct grid_scan_default_items_per_thread
{
  static constexpr size_t value =
      grid_scan_max_items_per_thread<DataType, block_size>::value / 2;
};

// tuning for gfx90a
template < typename DataType, size_t block_size >
struct grid_scan_default_items_per_thread<
    DataType, block_size, 910, std::enable_if_t<sizeof(DataType) == sizeof(double)> >
{
  static constexpr size_t value =
      (block_size <= 64) ? 6 :
      (block_size <= 128) ? 4 :
      (block_size <= 256) ? 4 :
      (block_size <= 512) ? 4 :
      (block_size <= 1024) ? 2 : 1;
};

// tuning for gfx942
template < typename DataType, size_t block_size >
struct grid_scan_default_items_per_thread<
    DataType, block_size, 942, std::enable_if_t<sizeof(DataType) == sizeof(double)>>
{
  static constexpr size_t value =
      (block_size <= 64) ? 22 :
      (block_size <= 128) ? 22 :
      (block_size <= 256) ? 19 :
      (block_size <= 512) ? 13 :
      (block_size <= 1024) ? 7 : 1;
};

} // end namespace hip
} // end namespace detail
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
