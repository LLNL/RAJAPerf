//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <rocprim/block/block_reduce.hpp>
#include <rocprim/warp/warp_reduce.hpp>

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

const size_t warp_size = detail::hip::warp_size;

static constexpr size_t atomic_stride =
    (detail::hip::cache_line_size > sizeof(ATOMIC::Data_type))
    ? detail::hip::cache_line_size / sizeof(ATOMIC::Data_type)
    : 1;

static constexpr size_t max_concurrent_atomics =
    (detail::hip::max_concurrent_atomic_bytes > sizeof(ATOMIC::Data_type))
    ? detail::hip::max_concurrent_atomic_bytes / sizeof(ATOMIC::Data_type)
    : 1;

using atomic_orderings = camp::list<
    detail::GetModReorderStatic,
    detail::GetStridedReorderStatic<atomic_stride, max_concurrent_atomics> >;


template < size_t block_size, typename AtomicOrdering >
__launch_bounds__(block_size)
__global__ void atomic_replicate_thread(ATOMIC::Data_ptr atomic,
                          Index_type iend,
                          AtomicOrdering atomic_ordering)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    ATOMIC_RAJA_BODY_DIRECT(RAJA::hip_atomic, atomic_ordering(i), ATOMIC_VALUE);
  }
}

template < size_t block_size, typename AtomicOrdering >
__launch_bounds__(block_size)
__global__ void atomic_replicate_warp(ATOMIC::Data_ptr atomic,
                          Index_type iend,
                          AtomicOrdering atomic_ordering)
{
  ATOMIC::Data_type val = 0;

  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    val = ATOMIC_VALUE;
  }

  using WarpReduce = rocprim::warp_reduce<ATOMIC::Data_type, warp_size>;
  __shared__ typename WarpReduce::storage_type warp_reduce_storage;
  WarpReduce().reduce(val, val, warp_reduce_storage);
  if ((threadIdx.x % warp_size) == 0) {
    ATOMIC_RAJA_BODY_DIRECT(RAJA::hip_atomic, atomic_ordering(i/warp_size), val);
  }
}

template < size_t block_size, typename AtomicOrdering >
__launch_bounds__(block_size)
__global__ void atomic_replicate_block(ATOMIC::Data_ptr atomic,
                          Index_type iend,
                          AtomicOrdering atomic_ordering)
{
  ATOMIC::Data_type val = 0;

  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    val = ATOMIC_VALUE;
  }

  using BlockReduce = rocprim::block_reduce<ATOMIC::Data_type, block_size>;
  __shared__ typename BlockReduce::storage_type block_reduce_storage;
  BlockReduce().reduce(val, val, block_reduce_storage);
  if (threadIdx.x == 0) {
    ATOMIC_RAJA_BODY_DIRECT(RAJA::hip_atomic, atomic_ordering(blockIdx.x), val);
  }
}


template < size_t block_size, typename AtomicOrdering >
void ATOMIC::runHipVariantReplicateGlobal(VariantID vid, AtomicOrdering atomic_ordering)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  const size_t allocation_size = atomic_ordering.range(iend);

  auto res{getHipResource()};

  ATOMIC_DATA_SETUP(allocation_size);

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (atomic_replicate_thread<block_size, AtomicOrdering>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         atomic,
                         iend,
                         atomic_ordering );

    }
    stopTimer();

  } else  if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::hip_exec<block_size, true /*async*/>>(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ATOMIC_RAJA_BODY_DIRECT(RAJA::hip_atomic, atomic_ordering(i), ATOMIC_VALUE);
      });

    }
    stopTimer();

  } else {
     getCout() << "\n  ATOMIC : Unknown Hip variant id = " << vid << std::endl;
  }

  ATOMIC_DATA_TEARDOWN(allocation_size);
}

template < size_t block_size, typename AtomicOrdering >
void ATOMIC::runHipVariantReplicateWarp(VariantID vid, AtomicOrdering atomic_ordering)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
  const size_t num_warps = grid_size*RAJA_DIVIDE_CEILING_INT(block_size, warp_size);
  const size_t allocation_size = atomic_ordering.range(num_warps);

  auto res{getHipResource()};

  ATOMIC_DATA_SETUP(allocation_size);

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (atomic_replicate_warp<block_size, AtomicOrdering>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         atomic,
                         iend,
                         atomic_ordering );

    }
    stopTimer();

  } else {
     getCout() << "\n  ATOMIC : Unknown Hip variant id = " << vid << std::endl;
  }

  ATOMIC_DATA_TEARDOWN(allocation_size);
}

template < size_t block_size, typename AtomicOrdering >
void ATOMIC::runHipVariantReplicateBlock(VariantID vid, AtomicOrdering atomic_ordering)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
  const size_t allocation_size = atomic_ordering.range(grid_size);

  auto res{getHipResource()};

  ATOMIC_DATA_SETUP(allocation_size);

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (atomic_replicate_block<block_size, AtomicOrdering>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         atomic,
                         iend,
                         atomic_ordering );

    }
    stopTimer();

  } else {
     getCout() << "\n  ATOMIC : Unknown Hip variant id = " << vid << std::endl;
  }

  ATOMIC_DATA_TEARDOWN(allocation_size);
}

void ATOMIC::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_HIP || vid == RAJA_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(atomic_orderings{}, [&](auto get_atomic_ordering) {

          seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

            if (run_params.numValidAtomicReplication() == 0u ||
                run_params.validAtomicReplication(replication)) {

              if (tune_idx == t) {

                setBlockSize(block_size);
                typename decltype(get_atomic_ordering)::template type<replication> atomic_ordering;
                runHipVariantReplicateGlobal<block_size>(vid, atomic_ordering);

              }

              t += 1;

            }

          });

          if ( vid == Base_HIP ) {

            seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

              if (run_params.numValidAtomicReplication() == 0u ||
                  run_params.validAtomicReplication(replication)) {

                if (tune_idx == t) {

                  setBlockSize(block_size);
                  typename decltype(get_atomic_ordering)::template type<replication> atomic_ordering;
                  runHipVariantReplicateWarp<block_size>(vid, atomic_ordering);

                }

                t += 1;

              }

            });

            seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

              if (run_params.numValidAtomicReplication() == 0u ||
                  run_params.validAtomicReplication(replication)) {

                if (tune_idx == t) {

                  setBlockSize(block_size);
                  typename decltype(get_atomic_ordering)::template type<replication> atomic_ordering;
                  runHipVariantReplicateBlock<block_size>(vid, atomic_ordering);

                }

                t += 1;

              }

            });

          }

        });

      }

    });

  } else {

    getCout() << "\n  ATOMIC : Unknown Hip variant id = " << vid << std::endl;

  }

}

void ATOMIC::setHipTuningDefinitions(VariantID vid)
{
  if ( vid == Base_HIP || vid == RAJA_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(atomic_orderings{}, [&](auto get_atomic_ordering) {

          seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

            if (run_params.numValidAtomicReplication() == 0u ||
                run_params.validAtomicReplication(replication)) {

              addVariantTuningName(vid, "replicate_"+std::to_string(replication)+
                                        "_order_"+get_atomic_ordering.name()+
                                        "_global_"+std::to_string(block_size));

            }

          });

          if ( vid == Base_HIP ) {

            seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

              if (run_params.numValidAtomicReplication() == 0u ||
                  run_params.validAtomicReplication(replication)) {

                addVariantTuningName(vid, "replicate_"+std::to_string(replication)+
                                          "_order_"+get_atomic_ordering.name()+
                                          "_warp_"+std::to_string(block_size));

              }

            });

            seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

              if (run_params.numValidAtomicReplication() == 0u ||
                  run_params.validAtomicReplication(replication)) {

                addVariantTuningName(vid, "replicate_"+std::to_string(replication)+
                                          "_order_"+get_atomic_ordering.name()+
                                          "_block_"+std::to_string(block_size));

              }

            });

          }

        });

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
