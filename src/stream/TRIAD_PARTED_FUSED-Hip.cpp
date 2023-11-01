//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD_PARTED_FUSED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"
#include "common/MemPool.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_SOA_SETUP_HIP \
  Index_type* len_ptrs; \
  Real_ptr*   a_ptrs; \
  Real_ptr*   b_ptrs; \
  Real_ptr*   c_ptrs; \
  Real_type*  alpha_ptrs; \
  Index_type* ibegin_ptrs; \
  allocData(DataSpace::HipPinnedCoarse, len_ptrs, parts.size()-1); \
  allocData(DataSpace::HipPinnedCoarse, a_ptrs, parts.size()-1); \
  allocData(DataSpace::HipPinnedCoarse, b_ptrs, parts.size()-1); \
  allocData(DataSpace::HipPinnedCoarse, c_ptrs, parts.size()-1); \
  allocData(DataSpace::HipPinnedCoarse, alpha_ptrs, parts.size()-1); \
  allocData(DataSpace::HipPinnedCoarse, ibegin_ptrs, parts.size()-1);

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_SOA_TEARDOWN_HIP \
  deallocData(DataSpace::HipPinnedCoarse, len_ptrs); \
  deallocData(DataSpace::HipPinnedCoarse, a_ptrs); \
  deallocData(DataSpace::HipPinnedCoarse, b_ptrs); \
  deallocData(DataSpace::HipPinnedCoarse, c_ptrs); \
  deallocData(DataSpace::HipPinnedCoarse, alpha_ptrs); \
  deallocData(DataSpace::HipPinnedCoarse, ibegin_ptrs);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void triad_parted_fused_soa(Index_type* len_ptrs, Real_ptr* a_ptrs,
                                       Real_ptr* b_ptrs, Real_ptr* c_ptrs,
                                       Real_type* alpha_ptrs, Index_type* ibegin_ptrs)
{
  Index_type j = blockIdx.y;

  Index_type len    = len_ptrs[j];
  Real_ptr   a      = a_ptrs[j];
  Real_ptr   b      = b_ptrs[j];
  Real_ptr   c      = c_ptrs[j];
  Real_type  alpha  = alpha_ptrs[j];
  Index_type ibegin = ibegin_ptrs[j];

  for (Index_type ii = threadIdx.x + blockIdx.x * block_size;
       ii < len;
       ii += block_size * gridDim.x) {
    Index_type i = ii + ibegin;
    TRIAD_PARTED_FUSED_BODY;
  }
}


#define TRIAD_PARTED_FUSED_MANUAL_FUSER_AOS_SETUP_HIP(num_holders) \
  triad_holder* triad_holders; \
  allocData(DataSpace::HipPinnedCoarse, triad_holders, (num_holders));

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_AOS_TEARDOWN_HIP \
  deallocData(DataSpace::HipPinnedCoarse, triad_holders);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void triad_parted_fused_aos(triad_holder* triad_holders)
{
  Index_type j = blockIdx.y;

  Index_type len    = triad_holders[j].len;
  Real_ptr   a      = triad_holders[j].a;
  Real_ptr   b      = triad_holders[j].b;
  Real_ptr   c      = triad_holders[j].c;
  Real_type  alpha  = triad_holders[j].alpha;
  Index_type ibegin = triad_holders[j].ibegin;

  for (Index_type ii = threadIdx.x + blockIdx.x * block_size;
       ii < len;
       ii += block_size * gridDim.x) {
    Index_type i = ii + ibegin;
    TRIAD_PARTED_FUSED_BODY;
  }
}


template < size_t block_size >
void TRIAD_PARTED_FUSED::runHipVariantSOASync(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  TRIAD_PARTED_FUSED_DATA_SETUP;

  if ( vid == Base_HIP ) {

    TRIAD_PARTED_FUSED_MANUAL_FUSER_SOA_SETUP_HIP

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Index_type index = 0;
      Index_type len_sum = 0;

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        len_ptrs[index] = iend-ibegin;
        a_ptrs[index] = a;
        b_ptrs[index] = b;
        c_ptrs[index] = c;
        alpha_ptrs[index] = alpha;
        ibegin_ptrs[index] = ibegin;
        len_sum += iend-ibegin;
        index += 1;
      }
      Index_type len_ave = (len_sum + index-1) / index;
      dim3 nthreads_per_block(block_size);
      dim3 nblocks((len_ave + block_size-1) / block_size, index);
      constexpr size_t shmem = 0;
      triad_parted_fused_soa<block_size><<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(
          len_ptrs, a_ptrs, b_ptrs, c_ptrs, alpha_ptrs, ibegin_ptrs);
      hipErrchk( hipGetLastError() );
      hipErrchk( hipStreamSynchronize( res.get_stream() ) );

    }
    stopTimer();

    TRIAD_PARTED_FUSED_MANUAL_FUSER_SOA_TEARDOWN_HIP

  } else {
      getCout() << "\n  TRIAD_PARTED_FUSED : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void TRIAD_PARTED_FUSED::runHipVariantAOSSync(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  TRIAD_PARTED_FUSED_DATA_SETUP;

  if ( vid == Base_HIP ) {

    const size_t num_holders = parts.size()-1;
    TRIAD_PARTED_FUSED_MANUAL_FUSER_AOS_SETUP_HIP(num_holders)

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Index_type index = 0;
      Index_type len_sum = 0;

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        triad_holders[index] = triad_holder{iend-ibegin, a, b, c, alpha, ibegin};
        len_sum += iend-ibegin;
        index += 1;
      }

      Index_type len_ave = (len_sum + index-1) / index;
      dim3 nthreads_per_block(block_size);
      dim3 nblocks((len_ave + block_size-1) / block_size, index);
      constexpr size_t shmem = 0;
      triad_parted_fused_aos<block_size><<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(
          triad_holders);
      hipErrchk( hipGetLastError() );
      hipErrchk( hipStreamSynchronize( res.get_stream() ) );

    }
    stopTimer();

    TRIAD_PARTED_FUSED_MANUAL_FUSER_AOS_TEARDOWN_HIP

  } else if ( vid == RAJA_HIP ) {

    auto triad_parted_fused_lam = [=] __device__ (Index_type i) {
          TRIAD_PARTED_FUSED_BODY;
        };

    using AllocatorHolder = RAJAPoolAllocatorHolder<RAJA::hip::pinned_mempool_type>;
    using Allocator = AllocatorHolder::Allocator<char>;

    AllocatorHolder allocatorHolder;

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::hip_work_async<block_size>,
                                 RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average,
                                 RAJA::constant_stride_array_of_objects,
                                 // RAJA::indirect_function_call_dispatch
                                 // RAJA::indirect_virtual_function_dispatch
                                 RAJA::direct_dispatch<camp::list<RAJA::TypedRangeSegment<Index_type>, decltype(triad_parted_fused_lam)>>
                                >;

    using workpool = RAJA::WorkPool< workgroup_policy,
                                     Index_type,
                                     RAJA::xargs<>,
                                     Allocator >;

    using workgroup = RAJA::WorkGroup< workgroup_policy,
                                       Index_type,
                                       RAJA::xargs<>,
                                       Allocator >;

    using worksite = RAJA::WorkSite< workgroup_policy,
                                     Index_type,
                                     RAJA::xargs<>,
                                     Allocator >;

    workpool pool(allocatorHolder.template getAllocator<char>());
    pool.reserve(parts.size()-1, 1024ull*1024ull);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        pool.enqueue(
            RAJA::TypedRangeSegment<Index_type>(ibegin, iend),
            triad_parted_fused_lam );
      }
      workgroup group = pool.instantiate();
      worksite site = group.run(res);
      res.wait();

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED_FUSED : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void TRIAD_PARTED_FUSED::runHipVariantAOSPoolSync(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  const size_t pool_size = 32ull * 1024ull * 1024ull;

  TRIAD_PARTED_FUSED_DATA_SETUP;

  if ( vid == Base_HIP ) {

    const size_t num_holders = std::max(parts.size()-1, pool_size / sizeof(triad_holder));
    TRIAD_PARTED_FUSED_MANUAL_FUSER_AOS_SETUP_HIP(num_holders)

    Index_type holder_start = 0;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      if (holder_start+parts.size()-1 > num_holders) {
        // synchronize when have to reuse memory
        hipErrchk( hipStreamSynchronize( res.get_stream() ) );
        holder_start = 0;
      }

      Index_type num_fused = 0;
      Index_type len_sum = 0;

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        triad_holders[holder_start+num_fused] = triad_holder{iend-ibegin, a, b, c, alpha, ibegin};
        len_sum += iend-ibegin;
        num_fused += 1;
      }

      Index_type len_ave = (len_sum + num_fused-1) / num_fused;
      dim3 nthreads_per_block(block_size);
      dim3 nblocks((len_ave + block_size-1) / block_size, num_fused);
      constexpr size_t shmem = 0;
      triad_parted_fused_aos<block_size><<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(
          triad_holders+holder_start);
      hipErrchk( hipGetLastError() );
      holder_start += num_fused;

    }
    stopTimer();

    TRIAD_PARTED_FUSED_MANUAL_FUSER_AOS_TEARDOWN_HIP

  } else if ( vid == RAJA_HIP ) {

    auto triad_parted_fused_lam = [=] __device__ (Index_type i) {
          TRIAD_PARTED_FUSED_BODY;
        };

    using AllocatorHolder = RAJAPoolAllocatorHolder<rajaperf::basic_mempool::MemPool<RAJA::hip::PinnedAllocator, camp::resources::Hip>>;
    using Allocator = AllocatorHolder::Allocator<char>;

    AllocatorHolder allocatorHolder(pool_size, res);

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::hip_work_async<block_size>,
                                 RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average,
                                 RAJA::constant_stride_array_of_objects,
                                 // RAJA::indirect_function_call_dispatch
                                 // RAJA::indirect_virtual_function_dispatch
                                 RAJA::direct_dispatch<camp::list<RAJA::TypedRangeSegment<Index_type>, decltype(triad_parted_fused_lam)>>
                                >;

    using workpool = RAJA::WorkPool< workgroup_policy,
                                     Index_type,
                                     RAJA::xargs<>,
                                     Allocator >;

    using workgroup = RAJA::WorkGroup< workgroup_policy,
                                       Index_type,
                                       RAJA::xargs<>,
                                       Allocator >;

    using worksite = RAJA::WorkSite< workgroup_policy,
                                     Index_type,
                                     RAJA::xargs<>,
                                     Allocator >;

    workpool pool(allocatorHolder.template getAllocator<char>());
    pool.reserve(parts.size()-1, 1024ull*1024ull);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        pool.enqueue(
            RAJA::TypedRangeSegment<Index_type>(ibegin, iend),
            triad_parted_fused_lam );
      }
      workgroup group = pool.instantiate();
      worksite site = group.run(res);

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED_FUSED : Unknown Hip variant id = " << vid << std::endl;
  }
}


template < size_t block_size >
void TRIAD_PARTED_FUSED::runHipVariantAOSReuse(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  TRIAD_PARTED_FUSED_DATA_SETUP;

  if ( vid == Base_HIP ) {

    const size_t num_holders = parts.size()-1;
    TRIAD_PARTED_FUSED_MANUAL_FUSER_AOS_SETUP_HIP(num_holders)

    Index_type index = 0;
    Index_type len_sum = 0;

    for (size_t p = 1; p < parts.size(); ++p ) {
      const Index_type ibegin = parts[p-1];
      const Index_type iend = parts[p];

      triad_holders[index] = triad_holder{iend-ibegin, a, b, c, alpha, ibegin};
      len_sum += iend-ibegin;
      index += 1;
    }
    Index_type len_ave = (len_sum + index-1) / index;
    dim3 nthreads_per_block(block_size);
    dim3 nblocks((len_ave + block_size-1) / block_size, index);
    constexpr size_t shmem = 0;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      triad_parted_fused_aos<block_size><<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(
          triad_holders);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    TRIAD_PARTED_FUSED_MANUAL_FUSER_AOS_TEARDOWN_HIP

  } else if ( vid == RAJA_HIP ) {

    auto triad_parted_fused_lam = [=] __device__ (Index_type i) {
          TRIAD_PARTED_FUSED_BODY;
        };

    using AllocatorHolder = RAJAPoolAllocatorHolder<RAJA::hip::pinned_mempool_type>;
    using Allocator = AllocatorHolder::Allocator<char>;

    AllocatorHolder allocatorHolder;

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::hip_work_async<block_size>,
                                 RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average,
                                 RAJA::constant_stride_array_of_objects,
                                 // RAJA::indirect_function_call_dispatch
                                 // RAJA::indirect_virtual_function_dispatch
                                 RAJA::direct_dispatch<camp::list<RAJA::TypedRangeSegment<Index_type>, decltype(triad_parted_fused_lam)>>
                                >;

    using workpool = RAJA::WorkPool< workgroup_policy,
                                     Index_type,
                                     RAJA::xargs<>,
                                     Allocator >;

    using workgroup = RAJA::WorkGroup< workgroup_policy,
                                       Index_type,
                                       RAJA::xargs<>,
                                       Allocator >;

    using worksite = RAJA::WorkSite< workgroup_policy,
                                     Index_type,
                                     RAJA::xargs<>,
                                     Allocator >;

    workpool pool(allocatorHolder.template getAllocator<char>());
    pool.reserve(parts.size()-1, 1024ull*1024ull);

    for (size_t p = 1; p < parts.size(); ++p ) {
      const Index_type ibegin = parts[p-1];
      const Index_type iend = parts[p];

      pool.enqueue(
          RAJA::TypedRangeSegment<Index_type>(ibegin, iend),
          triad_parted_fused_lam );
    }
    workgroup group = pool.instantiate();

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      worksite site = group.run(res);

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED_FUSED : Unknown Hip variant id = " << vid << std::endl;
  }
}

void TRIAD_PARTED_FUSED::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_HIP || vid == RAJA_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if ( vid == Base_HIP ) {

          if (tune_idx == t) {

            setBlockSize(block_size);
            runHipVariantSOASync<block_size>(vid);

          }

          t += 1;
        }

        if (tune_idx == t) {

          setBlockSize(block_size);
          runHipVariantAOSSync<block_size>(vid);

        }

        t += 1;

        if (tune_idx == t) {

          setBlockSize(block_size);
          runHipVariantAOSPoolSync<block_size>(vid);

        }

        t += 1;

        if (tune_idx == t) {

          setBlockSize(block_size);
          runHipVariantAOSReuse<block_size>(vid);

        }

        t += 1;

      }

    });

  } else {

    getCout() << "\n  TRIAD_PARTED_FUSED : Unknown Hip variant id = " << vid << std::endl;

  }

}

void TRIAD_PARTED_FUSED::setHipTuningDefinitions(VariantID vid)
{
  if ( vid == Base_HIP || vid == RAJA_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if ( vid == Base_HIP ) {
          addVariantTuningName(vid, "SOAsync_"+std::to_string(block_size));
        }

        addVariantTuningName(vid, "AOSsync_"+std::to_string(block_size));

        addVariantTuningName(vid, "AOSpoolsync_"+std::to_string(block_size));

        addVariantTuningName(vid, "AOSreuse_"+std::to_string(block_size));

      }

    });

  }
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
