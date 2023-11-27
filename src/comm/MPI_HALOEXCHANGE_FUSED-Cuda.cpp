//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MPI_HALOEXCHANGE_FUSED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI) && defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace comm
{

#define MPI_HALOEXCHANGE_FUSED_MANUAL_FUSER_SETUP_CUDA \
  Real_ptr*   pack_buffer_ptrs; \
  Int_ptr*    pack_list_ptrs; \
  Real_ptr*   pack_var_ptrs; \
  Index_type* pack_len_ptrs; \
  allocData(DataSpace::CudaPinned, pack_buffer_ptrs, num_neighbors * num_vars); \
  allocData(DataSpace::CudaPinned, pack_list_ptrs,   num_neighbors * num_vars); \
  allocData(DataSpace::CudaPinned, pack_var_ptrs,    num_neighbors * num_vars); \
  allocData(DataSpace::CudaPinned, pack_len_ptrs,    num_neighbors * num_vars); \
  Real_ptr*   unpack_buffer_ptrs; \
  Int_ptr*    unpack_list_ptrs; \
  Real_ptr*   unpack_var_ptrs; \
  Index_type* unpack_len_ptrs; \
  allocData(DataSpace::CudaPinned, unpack_buffer_ptrs, num_neighbors * num_vars); \
  allocData(DataSpace::CudaPinned, unpack_list_ptrs,   num_neighbors * num_vars); \
  allocData(DataSpace::CudaPinned, unpack_var_ptrs,    num_neighbors * num_vars); \
  allocData(DataSpace::CudaPinned, unpack_len_ptrs,    num_neighbors * num_vars);

#define MPI_HALOEXCHANGE_FUSED_MANUAL_FUSER_TEARDOWN_CUDA \
  deallocData(DataSpace::CudaPinned, pack_buffer_ptrs); \
  deallocData(DataSpace::CudaPinned, pack_list_ptrs); \
  deallocData(DataSpace::CudaPinned, pack_var_ptrs); \
  deallocData(DataSpace::CudaPinned, pack_len_ptrs); \
  deallocData(DataSpace::CudaPinned, unpack_buffer_ptrs); \
  deallocData(DataSpace::CudaPinned, unpack_list_ptrs); \
  deallocData(DataSpace::CudaPinned, unpack_var_ptrs); \
  deallocData(DataSpace::CudaPinned, unpack_len_ptrs);

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
    HALO_PACK_BODY;
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
    HALO_UNPACK_BODY;
  }
}


template < size_t block_size >
void MPI_HALOEXCHANGE_FUSED::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  MPI_HALOEXCHANGE_FUSED_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    MPI_HALOEXCHANGE_FUSED_MANUAL_FUSER_SETUP_CUDA;

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
      haloexchange_fused_pack<block_size><<<pack_nblocks, pack_nthreads_per_block, shmem, res.get_stream()>>>(
          pack_buffer_ptrs, pack_list_ptrs, pack_var_ptrs, pack_len_ptrs);
      cudaErrchk( cudaGetLastError() );
      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );
      if (separate_buffers) {
        for (Index_type l = 0; l < num_neighbors; ++l) {
          Index_type len = pack_index_list_lengths[l];
          copyData(DataSpace::Host, send_buffers[l],
                   dataSpace, pack_buffers[l],
                   len*num_vars);
        }
      }
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
        Index_type  len  = unpack_index_list_lengths[l];
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
      haloexchange_fused_unpack<block_size><<<unpack_nblocks, unpack_nthreads_per_block, shmem, res.get_stream()>>>(
          unpack_buffer_ptrs, unpack_list_ptrs, unpack_var_ptrs, unpack_len_ptrs);
      cudaErrchk( cudaGetLastError() );
      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );

      MPI_Waitall(num_neighbors, pack_mpi_requests.data(), MPI_STATUSES_IGNORE);

    }
    stopTimer();

    MPI_HALOEXCHANGE_FUSED_MANUAL_FUSER_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    using AllocatorHolder = RAJAPoolAllocatorHolder<RAJA::cuda::pinned_mempool_type>;
    using Allocator = AllocatorHolder::Allocator<char>;

    AllocatorHolder allocatorHolder;

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::cuda_work_async<block_size>,
                                 RAJA::unordered_cuda_loop_y_block_iter_x_threadblock_average,
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
          auto haloexchange_fused_pack_base_lam = [=] __device__ (Index_type i) {
                HALO_PACK_BODY;
              };
          pool_pack.enqueue(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_fused_pack_base_lam );
          buffer += len;
        }
      }
      workgroup group_pack = pool_pack.instantiate();
      worksite site_pack = group_pack.run(res);
      res.wait();
      if (separate_buffers) {
        for (Index_type l = 0; l < num_neighbors; ++l) {
          Index_type len = pack_index_list_lengths[l];
          copyData(DataSpace::Host, send_buffers[l],
                   dataSpace, pack_buffers[l],
                   len*num_vars);
        }
      }
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
          auto haloexchange_fused_unpack_base_lam = [=] __device__ (Index_type i) {
                HALO_UNPACK_BODY;
              };
          pool_unpack.enqueue(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_fused_unpack_base_lam );
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
     getCout() << "\n MPI_HALOEXCHANGE_FUSED : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MPI_HALOEXCHANGE_FUSED, Cuda)

} // end namespace comm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA