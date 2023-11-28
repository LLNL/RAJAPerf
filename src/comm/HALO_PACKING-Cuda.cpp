//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALO_PACKING.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace comm
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void HALO_exchange_pack(Real_ptr buffer, Int_ptr list, Real_ptr var,
                                  Index_type len)
{
   Index_type i = threadIdx.x + blockIdx.x * block_size;

   if (i < len) {
     HALO_PACK_BODY;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void HALO_exchange_unpack(Real_ptr buffer, Int_ptr list, Real_ptr var,
                                    Index_type len)
{
   Index_type i = threadIdx.x + blockIdx.x * block_size;

   if (i < len) {
     HALO_UNPACK_BODY;
   }
}


template < size_t block_size >
void HALO_PACKING::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  HALO_PACKING_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = pack_buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type  len  = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          dim3 nthreads_per_block(block_size);
          dim3 nblocks((len + block_size-1) / block_size);
          constexpr size_t shmem = 0;
          HALO_exchange_pack<block_size><<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(buffer, list, var, len);
          cudaErrchk( cudaGetLastError() );
          buffer += len;
        }

        if (separate_buffers) {
          copyData(DataSpace::Host, send_buffers[l],
                   dataSpace, pack_buffers[l],
                   len*num_vars);
        }

        cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );
      }

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
          dim3 nthreads_per_block(block_size);
          dim3 nblocks((len + block_size-1) / block_size);
          constexpr size_t shmem = 0;
          HALO_exchange_unpack<block_size><<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(buffer, list, var, len);
          cudaErrchk( cudaGetLastError() );
          buffer += len;
        }
      }
      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    using EXEC_POL = RAJA::cuda_exec<block_size, true /*async*/>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = pack_buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type  len  = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto HALO_exchange_pack_base_lam = [=] __device__ (Index_type i) {
                HALO_PACK_BODY;
              };
          RAJA::forall<EXEC_POL>( res,
              RAJA::TypedRangeSegment<Index_type>(0, len),
              HALO_exchange_pack_base_lam );
          buffer += len;
        }

        if (separate_buffers) {
          copyData(DataSpace::Host, send_buffers[l],
                   dataSpace, pack_buffers[l],
                   len*num_vars);
        }

        res.wait();
      }

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
          auto HALO_exchange_unpack_base_lam = [=] __device__ (Index_type i) {
                HALO_UNPACK_BODY;
              };
          RAJA::forall<EXEC_POL>( res,
              RAJA::TypedRangeSegment<Index_type>(0, len),
              HALO_exchange_unpack_base_lam );
          buffer += len;
        }
      }
      res.wait();

    }
    stopTimer();

  } else {
     getCout() << "\n HALO_PACKING : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(HALO_PACKING, Cuda)

} // end namespace comm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
