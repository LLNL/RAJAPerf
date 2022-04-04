//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALOEXCHANGE_FUSED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

#define HALOEXCHANGE_FUSED_DATA_SETUP_HIP \
  for (Index_type v = 0; v < m_num_vars; ++v) { \
    allocAndInitHipDeviceData(vars[v], m_vars[v], m_var_size); \
  } \
  for (Index_type l = 0; l < num_neighbors; ++l) { \
    allocAndInitHipDeviceData(buffers[l], m_buffers[l], m_num_vars*m_pack_index_list_lengths[l]); \
    allocAndInitHipDeviceData(pack_index_lists[l], m_pack_index_lists[l], m_pack_index_list_lengths[l]); \
    allocAndInitHipDeviceData(unpack_index_lists[l], m_unpack_index_lists[l], m_unpack_index_list_lengths[l]); \
  }

#define HALOEXCHANGE_FUSED_DATA_TEARDOWN_HIP \
  for (Index_type l = 0; l < num_neighbors; ++l) { \
    deallocHipDeviceData(unpack_index_lists[l]); \
    deallocHipDeviceData(pack_index_lists[l]); \
    deallocHipDeviceData(buffers[l]); \
  } \
  for (Index_type v = 0; v < m_num_vars; ++v) { \
    getHipDeviceData(m_vars[v], vars[v], m_var_size); \
    deallocHipDeviceData(vars[v]); \
  }

#define HALOEXCHANGE_FUSED_MANUAL_FUSER_SETUP_HIP \
  Real_ptr*   pack_buffer_ptrs; \
  Int_ptr*    pack_list_ptrs; \
  Real_ptr*   pack_var_ptrs; \
  Index_type* pack_len_ptrs; \
  allocHipPinnedData(pack_buffer_ptrs, num_neighbors * num_vars); \
  allocHipPinnedData(pack_list_ptrs,   num_neighbors * num_vars); \
  allocHipPinnedData(pack_var_ptrs,    num_neighbors * num_vars); \
  allocHipPinnedData(pack_len_ptrs,    num_neighbors * num_vars); \
  Real_ptr*   unpack_buffer_ptrs; \
  Int_ptr*    unpack_list_ptrs; \
  Real_ptr*   unpack_var_ptrs; \
  Index_type* unpack_len_ptrs; \
  allocHipPinnedData(unpack_buffer_ptrs, num_neighbors * num_vars); \
  allocHipPinnedData(unpack_list_ptrs,   num_neighbors * num_vars); \
  allocHipPinnedData(unpack_var_ptrs,    num_neighbors * num_vars); \
  allocHipPinnedData(unpack_len_ptrs,    num_neighbors * num_vars);

#define HALOEXCHANGE_FUSED_MANUAL_FUSER_TEARDOWN_HIP \
  deallocHipPinnedData(pack_buffer_ptrs); \
  deallocHipPinnedData(pack_list_ptrs); \
  deallocHipPinnedData(pack_var_ptrs); \
  deallocHipPinnedData(pack_len_ptrs); \
  deallocHipPinnedData(unpack_buffer_ptrs); \
  deallocHipPinnedData(unpack_list_ptrs); \
  deallocHipPinnedData(unpack_var_ptrs); \
  deallocHipPinnedData(unpack_len_ptrs);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void haloexchange_fused_pack(Real_ptr* pack_buffer_ptrs, Int_ptr* pack_list_ptrs,
                                        Real_ptr* pack_var_ptrs, Index_type* pack_len_ptrs)
{
  Index_type j = blockIdx.y;

  Real_ptr   buffer = pack_buffer_ptrs[j];
  Int_ptr    list   = pack_list_ptrs[j];
  Real_ptr   var    = pack_var_ptrs[j];
  Index_type len    = pack_len_ptrs[j];

  for (Index_type i = threadIdx.x + blockIdx.x * block_size;
       i < len;
       i += block_size * gridDim.x) {
    HALOEXCHANGE_FUSED_PACK_BODY;
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void haloexchange_fused_unpack(Real_ptr* unpack_buffer_ptrs, Int_ptr* unpack_list_ptrs,
                                          Real_ptr* unpack_var_ptrs, Index_type* unpack_len_ptrs)
{
  Index_type j = blockIdx.y;

  Real_ptr   buffer = unpack_buffer_ptrs[j];
  Int_ptr    list   = unpack_list_ptrs[j];
  Real_ptr   var    = unpack_var_ptrs[j];
  Index_type len    = unpack_len_ptrs[j];

  for (Index_type i = threadIdx.x + blockIdx.x * block_size;
       i < len;
       i += block_size * gridDim.x) {
    HALOEXCHANGE_FUSED_UNPACK_BODY;
  }
}


template < size_t block_size >
void HALOEXCHANGE_FUSED::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  HALOEXCHANGE_FUSED_DATA_SETUP;

  if ( vid == Base_HIP ) {

    HALOEXCHANGE_FUSED_DATA_SETUP_HIP;

    HALOEXCHANGE_FUSED_MANUAL_FUSER_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Index_type pack_index = 0;
      Index_type pack_len_sum = 0;

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type  len  = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          pack_buffer_ptrs[pack_index] = buffer;
          pack_list_ptrs[pack_index] = list;
          pack_var_ptrs[pack_index] = var;
          pack_len_ptrs[pack_index] = len;
          pack_len_sum += len;
          pack_index += 1;
          buffer += len;
        }
      }
      Index_type pack_len_ave = (pack_len_sum + pack_index-1) / pack_index;
      dim3 pack_nthreads_per_block(block_size);
      dim3 pack_nblocks((pack_len_ave + block_size-1) / block_size, pack_index);
      hipLaunchKernelGGL((haloexchange_fused_pack<block_size>), pack_nblocks, pack_nthreads_per_block, 0, 0,
          pack_buffer_ptrs, pack_list_ptrs, pack_var_ptrs, pack_len_ptrs);
      hipErrchk( hipGetLastError() );
      synchronize();

      Index_type unpack_index = 0;
      Index_type unpack_len_sum = 0;

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type  len  = unpack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          unpack_buffer_ptrs[unpack_index] = buffer;
          unpack_list_ptrs[unpack_index] = list;
          unpack_var_ptrs[unpack_index] = var;
          unpack_len_ptrs[unpack_index] = len;
          unpack_len_sum += len;
          unpack_index += 1;
          buffer += len;
        }
      }
      Index_type unpack_len_ave = (unpack_len_sum + unpack_index-1) / unpack_index;
      dim3 unpack_nthreads_per_block(block_size);
      dim3 unpack_nblocks((unpack_len_ave + block_size-1) / block_size, unpack_index);
      hipLaunchKernelGGL((haloexchange_fused_unpack<block_size>), unpack_nblocks, unpack_nthreads_per_block, 0, 0,
          unpack_buffer_ptrs, unpack_list_ptrs, unpack_var_ptrs, unpack_len_ptrs);
      hipErrchk( hipGetLastError() );
      synchronize();

    }
    stopTimer();

    HALOEXCHANGE_FUSED_MANUAL_FUSER_TEARDOWN_HIP;

    HALOEXCHANGE_FUSED_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    HALOEXCHANGE_FUSED_DATA_SETUP_HIP;

    using AllocatorHolder = RAJAPoolAllocatorHolder<RAJA::hip::pinned_mempool_type>;
    using Allocator = AllocatorHolder::Allocator<char>;

    AllocatorHolder allocatorHolder;

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::hip_work_async<block_size>,
#ifdef RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL
                                 RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average,
#else
                                 RAJA::ordered,
#endif
                                 RAJA::constant_stride_array_of_objects >;

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

    workpool pool_pack  (allocatorHolder.template getAllocator<char>());
    workpool pool_unpack(allocatorHolder.template getAllocator<char>());
    pool_pack.reserve(num_neighbors * num_vars, 1024ull*1024ull);
    pool_unpack.reserve(num_neighbors * num_vars, 1024ull*1024ull);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type len = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto haloexchange_fused_pack_base_lam = [=] __device__ (Index_type i) {
                HALOEXCHANGE_FUSED_PACK_BODY;
              };
          pool_pack.enqueue(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_fused_pack_base_lam );
          buffer += len;
        }
      }
      workgroup group_pack = pool_pack.instantiate();
      worksite site_pack = group_pack.run();
      synchronize();

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type len = unpack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto haloexchange_fused_unpack_base_lam = [=] __device__ (Index_type i) {
                HALOEXCHANGE_FUSED_UNPACK_BODY;
              };
          pool_unpack.enqueue(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_fused_unpack_base_lam );
          buffer += len;
        }
      }
      workgroup group_unpack = pool_unpack.instantiate();
      worksite site_unpack = group_unpack.run();
      synchronize();

    }
    stopTimer();

    HALOEXCHANGE_FUSED_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n HALOEXCHANGE_FUSED : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(HALOEXCHANGE_FUSED, Hip)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
