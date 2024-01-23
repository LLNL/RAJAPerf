//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALO_EXCHANGE_FUSED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI) && defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"
#include "common/MemPool.hpp"

#include <iostream>

namespace rajaperf
{
namespace comm
{

#define HALO_EXCHANGE_FUSED_MANUAL_FUSER_SETUP_HIP \
  Real_ptr*   pack_buffer_ptrs; \
  Int_ptr*    pack_list_ptrs; \
  Real_ptr*   pack_var_ptrs; \
  Index_type* pack_len_ptrs; \
  allocData(DataSpace::HipPinnedCoarse, pack_buffer_ptrs, num_neighbors * num_vars); \
  allocData(DataSpace::HipPinnedCoarse, pack_list_ptrs,   num_neighbors * num_vars); \
  allocData(DataSpace::HipPinnedCoarse, pack_var_ptrs,    num_neighbors * num_vars); \
  allocData(DataSpace::HipPinnedCoarse, pack_len_ptrs,    num_neighbors * num_vars); \
  Real_ptr*   unpack_buffer_ptrs; \
  Int_ptr*    unpack_list_ptrs; \
  Real_ptr*   unpack_var_ptrs; \
  Index_type* unpack_len_ptrs; \
  allocData(DataSpace::HipPinnedCoarse, unpack_buffer_ptrs, num_neighbors * num_vars); \
  allocData(DataSpace::HipPinnedCoarse, unpack_list_ptrs,   num_neighbors * num_vars); \
  allocData(DataSpace::HipPinnedCoarse, unpack_var_ptrs,    num_neighbors * num_vars); \
  allocData(DataSpace::HipPinnedCoarse, unpack_len_ptrs,    num_neighbors * num_vars);

#define HALO_EXCHANGE_FUSED_MANUAL_FUSER_TEARDOWN_HIP \
  deallocData(DataSpace::HipPinnedCoarse, pack_buffer_ptrs); \
  deallocData(DataSpace::HipPinnedCoarse, pack_list_ptrs); \
  deallocData(DataSpace::HipPinnedCoarse, pack_var_ptrs); \
  deallocData(DataSpace::HipPinnedCoarse, pack_len_ptrs); \
  deallocData(DataSpace::HipPinnedCoarse, unpack_buffer_ptrs); \
  deallocData(DataSpace::HipPinnedCoarse, unpack_list_ptrs); \
  deallocData(DataSpace::HipPinnedCoarse, unpack_var_ptrs); \
  deallocData(DataSpace::HipPinnedCoarse, unpack_len_ptrs);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void halo_exchange_fused_pack(Real_ptr* pack_buffer_ptrs, Int_ptr* pack_list_ptrs,
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
    HALO_PACK_BODY;
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void halo_exchange_fused_unpack(Real_ptr* unpack_buffer_ptrs, Int_ptr* unpack_list_ptrs,
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
    HALO_UNPACK_BODY;
  }
}


template < size_t block_size >
void HALO_EXCHANGE_FUSED::runHipVariantDirect(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  HALO_EXCHANGE_FUSED_DATA_SETUP;

  if ( vid == Base_HIP ) {

    HALO_EXCHANGE_FUSED_MANUAL_FUSER_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      constexpr size_t shmem = 0;

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Index_type len = unpack_index_list_lengths[l];
        MPI_Irecv(recv_buffers[l], len*num_vars, Real_MPI_type,
            mpi_ranks[l], recv_tags[l], MPI_COMM_WORLD, &unpack_mpi_requests[l]);
      }

      Index_type pack_index = 0;
      Index_type pack_len_sum = 0;

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = pack_buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type len = pack_index_list_lengths[l];
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
      RPlaunchHipKernel( (halo_exchange_fused_pack<block_size>),
                         pack_nblocks, pack_nthreads_per_block,
                         shmem, res.get_stream(),
                         pack_buffer_ptrs,
                         pack_list_ptrs,
                         pack_var_ptrs,
                         pack_len_ptrs);
      if (separate_buffers) {
        for (Index_type l = 0; l < num_neighbors; ++l) {
          Index_type len = pack_index_list_lengths[l];
          copyData(DataSpace::Host, send_buffers[l],
                   dataSpace, pack_buffers[l],
                   len*num_vars);
        }
      }
      hipErrchk( hipStreamSynchronize( res.get_stream() ) );
      for (Index_type l = 0; l < num_neighbors; ++l) {
        Index_type len = pack_index_list_lengths[l];
        MPI_Isend(send_buffers[l], len*num_vars, Real_MPI_type,
            mpi_ranks[l], send_tags[l], MPI_COMM_WORLD, &pack_mpi_requests[l]);
      }

      MPI_Waitall(num_neighbors, unpack_mpi_requests.data(), MPI_STATUSES_IGNORE);

      Index_type unpack_index = 0;
      Index_type unpack_len_sum = 0;

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = unpack_buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type len = unpack_index_list_lengths[l];
        if (separate_buffers) {
          copyData(dataSpace, unpack_buffers[l],
                   DataSpace::Host, recv_buffers[l],
                   len*num_vars);
        }

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
      RPlaunchHipKernel( (halo_exchange_fused_unpack<block_size>),
                         unpack_nblocks, unpack_nthreads_per_block,
                         shmem, res.get_stream(),
                         unpack_buffer_ptrs,
                         unpack_list_ptrs,
                         unpack_var_ptrs,
                         unpack_len_ptrs);
      hipErrchk( hipStreamSynchronize( res.get_stream() ) );

      MPI_Waitall(num_neighbors, pack_mpi_requests.data(), MPI_STATUSES_IGNORE);

    }
    stopTimer();

    HALO_EXCHANGE_FUSED_MANUAL_FUSER_TEARDOWN_HIP;

  } else {
     getCout() << "\n HALO_EXCHANGE_FUSED : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size, typename dispatch_helper >
void HALO_EXCHANGE_FUSED::runHipVariantWorkGroup(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  HALO_EXCHANGE_FUSED_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    using AllocatorHolder = RAJAPoolAllocatorHolder<
        rajaperf::basic_mempool::MemPool<dataspace_allocator<DataSpace::HipPinnedCoarse>>>;
    using Allocator = AllocatorHolder::Allocator<char>;

    AllocatorHolder allocatorHolder;

    using range_segment = RAJA::TypedRangeSegment<Index_type>;

    using dispatch_policy = typename dispatch_helper::template dispatch_policy<
                              camp::list<range_segment, Packer>,
                              camp::list<range_segment, UnPacker>>;

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::hip_work_async<block_size>,
                                 RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average,
                                 RAJA::constant_stride_array_of_objects,
                                 dispatch_policy >;

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
        Index_type len = unpack_index_list_lengths[l];
        MPI_Irecv(recv_buffers[l], len*num_vars, Real_MPI_type,
            mpi_ranks[l], recv_tags[l], MPI_COMM_WORLD, &unpack_mpi_requests[l]);
      }

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = pack_buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type len = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          pool_pack.enqueue(range_segment(0, len), Packer{buffer, var, list});
          buffer += len;
        }
      }
      workgroup group_pack = pool_pack.instantiate();
      worksite site_pack = group_pack.run(res);
      if (separate_buffers) {
        for (Index_type l = 0; l < num_neighbors; ++l) {
          Index_type len = pack_index_list_lengths[l];
          copyData(DataSpace::Host, send_buffers[l],
                   dataSpace, pack_buffers[l],
                   len*num_vars);
        }
      }
      res.wait();
      for (Index_type l = 0; l < num_neighbors; ++l) {
        Index_type len = pack_index_list_lengths[l];
        MPI_Isend(send_buffers[l], len*num_vars, Real_MPI_type,
            mpi_ranks[l], send_tags[l], MPI_COMM_WORLD, &pack_mpi_requests[l]);
      }

      MPI_Waitall(num_neighbors, unpack_mpi_requests.data(), MPI_STATUSES_IGNORE);

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = unpack_buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type len = unpack_index_list_lengths[l];
        if (separate_buffers) {
          copyData(dataSpace, unpack_buffers[l],
                   DataSpace::Host, recv_buffers[l],
                   len*num_vars);
        }

        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          pool_unpack.enqueue(range_segment(0, len), UnPacker{buffer, var, list});
          buffer += len;
        }
      }
      workgroup group_unpack = pool_unpack.instantiate();
      worksite site_unpack = group_unpack.run(res);
      res.wait();

      MPI_Waitall(num_neighbors, pack_mpi_requests.data(), MPI_STATUSES_IGNORE);

    }
    stopTimer();

  } else {
     getCout() << "\n HALO_EXCHANGE_FUSED : Unknown Hip variant id = " << vid << std::endl;
  }
}


void HALO_EXCHANGE_FUSED::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if (vid == Base_HIP) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (tune_idx == t) {

          runHipVariantDirect<block_size>(vid);

        }

        t += 1;

      }

    });

  }

  if (vid == RAJA_HIP) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(hip_workgroup_dispatch_helpers{}, [&](auto dispatch_helper) {

          if (tune_idx == t) {

            runHipVariantWorkGroup<decltype(block_size){}, decltype(dispatch_helper)>(vid);

          }

          t += 1;

        });

      }

    });

  }
}

void HALO_EXCHANGE_FUSED::setHipTuningDefinitions(VariantID vid)
{
  if (vid == Base_HIP) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        addVariantTuningName(vid, "direct_"+std::to_string(block_size));

      }

    });

  }

  if (vid == RAJA_HIP) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(hip_workgroup_dispatch_helpers{}, [&](auto dispatch_helper) {

          addVariantTuningName(vid, decltype(dispatch_helper)::get_name()+"_"+std::to_string(block_size));

        });

      }

    });

  }
}

} // end namespace comm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
