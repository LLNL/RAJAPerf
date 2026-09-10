//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALO_PACKING.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace comm
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void halo_packing_pack(Real_ptr buffer, Int_ptr list, Real_ptr var,
                                  Index_type len)
{
   Index_type i = threadIdx.x + blockIdx.x * block_size;

   if (i < len) {
     HALO_PACK_BODY;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void halo_packing_unpack(Real_ptr buffer, Int_ptr list, Real_ptr var,
                                    Index_type len)
{
   Index_type i = threadIdx.x + blockIdx.x * block_size;

   if (i < len) {
     HALO_UNPACK_BODY;
   }
}


template < size_t block_size >
void HALO_PACKING::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  HALO_PACKING_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = pack_buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type len = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          dim3 nthreads_per_block(block_size);
          dim3 nblocks((len + block_size-1) / block_size);
          constexpr size_t shmem = 0;
          RP_CALI_SUBKERNEL_BEGIN("HALO_PACKING_pack_k");
          RPlaunchHipKernel( (halo_packing_pack<block_size>),
                             nblocks, nthreads_per_block,
                             shmem, res.get_stream(),
                             buffer, list, var, len );
          RP_CALI_SUBKERNEL_END("HALO_PACKING_pack_k");
          buffer += len;
        }

        if (separate_buffers) {
          CAMP_HIP_API_INVOKE_AND_CHECK( hipMemcpyAsync,
              send_buffers[l], pack_buffers[l], len*num_vars*sizeof(Real_type),
              hipMemcpyDefault, res.get_stream() );
        }

        CAMP_HIP_API_INVOKE_AND_CHECK( hipStreamSynchronize, res.get_stream() );
      }

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = unpack_buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type len = unpack_index_list_lengths[l];
        if (separate_buffers) {
          CAMP_HIP_API_INVOKE_AND_CHECK( hipMemcpyAsync,
              unpack_buffers[l], recv_buffers[l], len*num_vars*sizeof(Real_type),
              hipMemcpyDefault, res.get_stream() );
        }

        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          dim3 nthreads_per_block(block_size);
          dim3 nblocks((len + block_size-1) / block_size);
          constexpr size_t shmem = 0;
          RP_CALI_SUBKERNEL_BEGIN("HALO_PACKING_unpack_k");
          RPlaunchHipKernel( (halo_packing_unpack<block_size>),
                             nblocks, nthreads_per_block,
                             shmem, res.get_stream(),
                             buffer, list, var, len );
          RP_CALI_SUBKERNEL_END("HALO_PACKING_unpack_k");
          buffer += len;
        }
      }
      CAMP_HIP_API_INVOKE_AND_CHECK( hipStreamSynchronize, res.get_stream() );

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    using EXEC_POL = RAJA::hip_exec<block_size, true /*async*/>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = pack_buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type len = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto halo_packing_pack_base_lam = [=] __device__ (Index_type i) {
                HALO_PACK_BODY;
              };
          RP_CALI_SUBKERNEL_BEGIN("HALO_PACKING_pack_k");
          RAJA::forall<EXEC_POL>( res,
              RAJA::TypedRangeSegment<Index_type>(0, len),
              halo_packing_pack_base_lam );
          RP_CALI_SUBKERNEL_END("HALO_PACKING_pack_k");
          buffer += len;
        }

        if (separate_buffers) {
          res.memcpy(send_buffers[l], pack_buffers[l], len*num_vars*sizeof(Real_type));
        }

        res.wait();
      }

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = unpack_buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type len = unpack_index_list_lengths[l];
        if (separate_buffers) {
          res.memcpy(unpack_buffers[l], recv_buffers[l], len*num_vars*sizeof(Real_type));
        }

        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto halo_packing_unpack_base_lam = [=] __device__ (Index_type i) {
                HALO_UNPACK_BODY;
              };
          RP_CALI_SUBKERNEL_BEGIN("HALO_PACKING_unpack_k");
          RAJA::forall<EXEC_POL>( res,
              RAJA::TypedRangeSegment<Index_type>(0, len),
              halo_packing_unpack_base_lam );
          RP_CALI_SUBKERNEL_END("HALO_PACKING_unpack_k");
          buffer += len;
        }
      }
      res.wait();

    }
    stopTimer();

  } else {
     getCout() << "\n HALO_PACKING : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(HALO_PACKING, Hip, Base_HIP, RAJA_HIP)

} // end namespace comm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
