//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MPI_HALOEXCHANGE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI) && defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void haloexchange_pack(Real_ptr buffer, Int_ptr list, Real_ptr var,
                                  Index_type len)
{
   Index_type i = threadIdx.x + blockIdx.x * block_size;

   if (i < len) {
     HALOEXCHANGE_PACK_BODY;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void haloexchange_unpack(Real_ptr buffer, Int_ptr list, Real_ptr var,
                                    Index_type len)
{
   Index_type i = threadIdx.x + blockIdx.x * block_size;

   if (i < len) {
     HALOEXCHANGE_UNPACK_BODY;
   }
}


template < size_t block_size >
void MPI_HALOEXCHANGE::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  MPI_HALOEXCHANGE_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Index_type len = unpack_index_list_lengths[l];
        int mpi_rank = mpi_ranks[l];
        MPI_Irecv(recv_buffers[l], len*num_vars, Real_MPI_type,
            mpi_rank, l, MPI_COMM_WORLD, &unpack_mpi_requests[l]);
      }

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = pack_buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type  len  = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          dim3 nthreads_per_block(block_size);
          dim3 nblocks((len + block_size-1) / block_size);
          haloexchange_pack<block_size><<<nblocks, nthreads_per_block>>>(buffer, list, var, len);
          cudaErrchk( cudaGetLastError() );
          buffer += len;
        }

        if (separate_buffers) {
          copyData(DataSpace::Host, send_buffers[l],
                   dataSpace, pack_buffers[l],
                   len*num_vars);
        }

        synchronize();
        int mpi_rank = mpi_ranks[l];
        MPI_Isend(send_buffers[l], len*num_vars, Real_MPI_type,
            mpi_rank, l, MPI_COMM_WORLD, &pack_mpi_requests[l]);
      }

      for (Index_type ll = 0; ll < num_neighbors; ++ll) {
        int l = -1;
        MPI_Waitany(num_neighbors, unpack_mpi_requests.data(), &l, MPI_STATUS_IGNORE);

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
          dim3 nthreads_per_block(block_size);
          dim3 nblocks((len + block_size-1) / block_size);
          haloexchange_unpack<block_size><<<nblocks, nthreads_per_block>>>(buffer, list, var, len);
          cudaErrchk( cudaGetLastError() );
          buffer += len;
        }
      }
      synchronize();

      MPI_Waitall(num_neighbors, pack_mpi_requests.data(), MPI_STATUSES_IGNORE);

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    using EXEC_POL = RAJA::cuda_exec<block_size, true /*async*/>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Index_type len = unpack_index_list_lengths[l];
        int mpi_rank = mpi_ranks[l];
        MPI_Irecv(recv_buffers[l], len*num_vars, Real_MPI_type,
            mpi_rank, l, MPI_COMM_WORLD, &unpack_mpi_requests[l]);
      }

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = pack_buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type  len  = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto haloexchange_pack_base_lam = [=] __device__ (Index_type i) {
                HALOEXCHANGE_PACK_BODY;
              };
          RAJA::forall<EXEC_POL>(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_pack_base_lam );
          buffer += len;
        }

        if (separate_buffers) {
          copyData(DataSpace::Host, send_buffers[l],
                   dataSpace, pack_buffers[l],
                   len*num_vars);
        }

        synchronize();
        int mpi_rank = mpi_ranks[l];
        MPI_Isend(send_buffers[l], len*num_vars, Real_MPI_type,
            mpi_rank, l, MPI_COMM_WORLD, &pack_mpi_requests[l]);
      }

      for (Index_type ll = 0; ll < num_neighbors; ++ll) {
        int l = -1;
        MPI_Waitany(num_neighbors, unpack_mpi_requests.data(), &l, MPI_STATUS_IGNORE);

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
          auto haloexchange_unpack_base_lam = [=] __device__ (Index_type i) {
                HALOEXCHANGE_UNPACK_BODY;
              };
          RAJA::forall<EXEC_POL>(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_unpack_base_lam );
          buffer += len;
        }
      }
      synchronize();

      MPI_Waitall(num_neighbors, pack_mpi_requests.data(), MPI_STATUSES_IGNORE);

    }
    stopTimer();

  } else {
     getCout() << "\n MPI_HALOEXCHANGE : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(MPI_HALOEXCHANGE, Cuda)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA