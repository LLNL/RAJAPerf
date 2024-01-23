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
  allocData(DataSpace::HipDevice, len_ptrs, parts.size()-1); \
  allocData(DataSpace::HipDevice, a_ptrs, parts.size()-1); \
  allocData(DataSpace::HipDevice, b_ptrs, parts.size()-1); \
  allocData(DataSpace::HipDevice, c_ptrs, parts.size()-1); \
  allocData(DataSpace::HipDevice, alpha_ptrs, parts.size()-1); \
  allocData(DataSpace::HipDevice, ibegin_ptrs, parts.size()-1);

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_SOA_TEARDOWN_HIP \
  deallocData(DataSpace::HipDevice, len_ptrs); \
  deallocData(DataSpace::HipDevice, a_ptrs); \
  deallocData(DataSpace::HipDevice, b_ptrs); \
  deallocData(DataSpace::HipDevice, c_ptrs); \
  deallocData(DataSpace::HipDevice, alpha_ptrs); \
  deallocData(DataSpace::HipDevice, ibegin_ptrs);

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
  allocData(DataSpace::HipDevice, triad_holders, (num_holders));

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_AOS_TEARDOWN_HIP \
  deallocData(DataSpace::HipDevice, triad_holders);

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

using scan_index_type = RAJA::hip_dim_member_t;
#define WARP_SIZE 64
#define warp_shfl(...) __shfl(__VA_ARGS__)

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void triad_parted_fused_scan_aos(scan_index_type* first_blocks, scan_index_type num_fused,
                                            triad_holder* triad_holders)
{
  scan_index_type min_j = 0;
  scan_index_type max_j = num_fused-1;
  scan_index_type j = (min_j + max_j + 1) / 2;
  scan_index_type first_block = first_blocks[j];
  while (min_j != max_j) {
    if (first_block > blockIdx.x) {
      max_j = j-1;
    } else {
      min_j = j;
    }
    j = (min_j + max_j + 1) / 2;
    first_block = first_blocks[j];
  }

  Index_type len    = triad_holders[j].len;
  Real_ptr   a      = triad_holders[j].a;
  Real_ptr   b      = triad_holders[j].b;
  Real_ptr   c      = triad_holders[j].c;
  Real_type  alpha  = triad_holders[j].alpha;
  Index_type ibegin = triad_holders[j].ibegin;

  Index_type ii = threadIdx.x + (blockIdx.x - first_block) * block_size;
  if (ii < len) {
    Index_type i = ii + ibegin;
    TRIAD_PARTED_FUSED_BODY;
  }
}


template < size_t block_size >
void TRIAD_PARTED_FUSED::runHipVariantSOA2dSync(VariantID vid)
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
void TRIAD_PARTED_FUSED::runHipVariantSOA2dReuse(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  TRIAD_PARTED_FUSED_DATA_SETUP;

  if ( vid == Base_HIP ) {

    TRIAD_PARTED_FUSED_MANUAL_FUSER_SOA_SETUP_HIP

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

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      triad_parted_fused_soa<block_size><<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(
          len_ptrs, a_ptrs, b_ptrs, c_ptrs, alpha_ptrs, ibegin_ptrs);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    TRIAD_PARTED_FUSED_MANUAL_FUSER_SOA_TEARDOWN_HIP

  } else {
      getCout() << "\n  TRIAD_PARTED_FUSED : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void TRIAD_PARTED_FUSED::runHipVariantAOS2dSync(VariantID vid)
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
void TRIAD_PARTED_FUSED::runHipVariantAOS2dPoolSync(VariantID vid)
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
void TRIAD_PARTED_FUSED::runHipVariantAOS2dReuse(VariantID vid)
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

template < size_t block_size >
void TRIAD_PARTED_FUSED::runHipVariantAOS2dReuseFunctionPointer(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  TRIAD_PARTED_FUSED_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

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
                                 RAJA::indirect_function_call_dispatch
                                 // RAJA::indirect_virtual_function_dispatch
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

template < size_t block_size >
void TRIAD_PARTED_FUSED::runHipVariantAOS2dReuseVirtualFunction(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  TRIAD_PARTED_FUSED_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

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
                                 RAJA::indirect_virtual_function_dispatch
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

template < size_t block_size >
void TRIAD_PARTED_FUSED::runHipVariantAOSScanReuse(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  TRIAD_PARTED_FUSED_DATA_SETUP;

  if ( vid == Base_HIP ) {

    const size_t num_holders = parts.size()-1;
    TRIAD_PARTED_FUSED_MANUAL_FUSER_AOS_SETUP_HIP(num_holders)
    scan_index_type* first_blocks;
    allocData(DataSpace::HipDevice, first_blocks, (num_holders));

    Index_type num_fused = 0;
    scan_index_type num_blocks = 0;

    for (size_t p = 1; p < parts.size(); ++p ) {
      const Index_type ibegin = parts[p-1];
      const Index_type iend = parts[p];

      triad_holders[num_fused] = triad_holder{iend-ibegin, a, b, c, alpha, ibegin};
      first_blocks[num_fused] = num_blocks;
      num_blocks += (static_cast<scan_index_type>(iend-ibegin) +
                     static_cast<scan_index_type>(block_size)-1) /
                    static_cast<scan_index_type>(block_size);
      num_fused += 1;
    }
    dim3 nthreads_per_block(block_size);
    dim3 nblocks(num_blocks);
    constexpr size_t shmem = 0;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      triad_parted_fused_scan_aos<block_size><<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(
          first_blocks, num_fused, triad_holders);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    deallocData(DataSpace::HipDevice, first_blocks);
    TRIAD_PARTED_FUSED_MANUAL_FUSER_AOS_TEARDOWN_HIP

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
            runHipVariantSOA2dSync<block_size>(vid);

          }

          t += 1;

          if (tune_idx == t) {

            setBlockSize(block_size);
            runHipVariantSOA2dReuse<block_size>(vid);

          }

          t += 1;

          if (tune_idx == t) {

            setBlockSize(block_size);
            runHipVariantAOSScanReuse<block_size>(vid);

          }

          t += 1;
        }

        if (tune_idx == t) {

          setBlockSize(block_size);
          runHipVariantAOS2dSync<block_size>(vid);

        }

        t += 1;

        if (tune_idx == t) {

          setBlockSize(block_size);
          runHipVariantAOS2dPoolSync<block_size>(vid);

        }

        t += 1;

        if (tune_idx == t) {

          setBlockSize(block_size);
          runHipVariantAOS2dReuse<block_size>(vid);

        }

        t += 1;

        if ( vid == RAJA_HIP ) {

          if (tune_idx == t) {

            setBlockSize(block_size);
            runHipVariantAOS2dReuseFunctionPointer<block_size>(vid);

          }

          t += 1;

          if (tune_idx == t) {

            setBlockSize(block_size);
            runHipVariantAOS2dReuseVirtualFunction<block_size>(vid);

          }

          t += 1;
        }

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
          addVariantTuningName(vid, "SOA_2d_sync_"+std::to_string(block_size));

          addVariantTuningName(vid, "SOA_2d_reuse_"+std::to_string(block_size));

          addVariantTuningName(vid, "AOS_scan_reuse_"+std::to_string(block_size));
        }

        addVariantTuningName(vid, "AOS_2d_sync_"+std::to_string(block_size));

        addVariantTuningName(vid, "AOS_2d_poolsync_"+std::to_string(block_size));

        addVariantTuningName(vid, "AOS_2d_reuse_"+std::to_string(block_size));

        if ( vid == RAJA_HIP ) {

          addVariantTuningName(vid, "AOS_2d_reuse_funcptr_"+std::to_string(block_size));
          addVariantTuningName(vid, "AOS_2d_reuse_virtfunc_"+std::to_string(block_size));

        }
      }

    });

  }
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
