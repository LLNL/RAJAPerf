//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST_3LOOP.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


template < size_t block_size >
__launch_bounds__(block_size)
__global__ void indexlist_conditional(Real_ptr x,
                                      Index_ptr counts,
                                      Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void indexlist_make_list(Int_ptr list,
                                    Index_ptr counts,
                                    Index_ptr len,
                                    Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    INDEXLIST_3LOOP_MAKE_LIST;
    if (i == iend-1) {
      *len = counts[i+1];
    }
  }
}


template < size_t block_size >
void INDEXLIST_3LOOP::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  INDEXLIST_3LOOP_DATA_SETUP;

  if ( vid == Base_HIP ) {

    INDEXLIST_3LOOP_COUNTS_SETUP(DataSpace::HipDevice);

    Index_ptr len;
    allocData(DataSpace::HipPinnedCoarse, len, 1);

    hipStream_t stream = res.get_stream();

    RAJA::operators::plus<Index_type> binary_op;
    Index_type init_val = 0;
    int scan_size = iend+1 - ibegin;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::exclusive_scan,
        d_temp_storage, temp_storage_bytes,
        counts+ibegin,
        counts+ibegin,
        init_val,
        scan_size,
        binary_op,
        stream);
#elif defined(__CUDACC__)
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
        d_temp_storage, temp_storage_bytes,
        counts+ibegin,
        counts+ibegin,
        binary_op,
        init_val,
        scan_size,
        stream);
#endif

    unsigned char* temp_storage;
    allocData(DataSpace::HipDevice, temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_3LOOP_1");
      RPlaunchHipKernel( (indexlist_conditional<block_size>),
                         grid_size, block_size,
                         shmem, stream,
                         x, counts, iend );
      RP_CALI_SUBKERNEL_END("INDEXLIST_3LOOP_1");

#if defined(__HIPCC__)
      CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::exclusive_scan,
          d_temp_storage, temp_storage_bytes,
          counts+ibegin,
          counts+ibegin,
          init_val,
          scan_size,
          binary_op,
          stream);
#elif defined(__CUDACC__)
      CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
          d_temp_storage, temp_storage_bytes,
          counts+ibegin,
          counts+ibegin,
          binary_op,
          init_val,
          scan_size,
          stream);
#endif

      RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_3LOOP_2");
      RPlaunchHipKernel( (indexlist_make_list<block_size>),
                         grid_size, block_size,
                         shmem, stream,
                         list, counts, len, iend );
      RP_CALI_SUBKERNEL_END("INDEXLIST_3LOOP_2");

      CAMP_HIP_API_INVOKE_AND_CHECK( hipStreamSynchronize, stream );
      m_len = *len;

    }
    stopTimer();

    deallocData(DataSpace::HipDevice, temp_storage);
    deallocData(DataSpace::HipPinnedCoarse, len);

    INDEXLIST_3LOOP_COUNTS_TEARDOWN(DataSpace::HipDevice);

  } else if ( vid == RAJA_HIP ) {

    INDEXLIST_3LOOP_COUNTS_SETUP(DataSpace::HipDevice);

    Index_ptr len;
    allocData(DataSpace::HipPinnedCoarse, len, 1);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_3LOOP_1");
      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend),
        [=] __device__ (Index_type i) {
        counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
      });
      RP_CALI_SUBKERNEL_END("INDEXLIST_3LOOP_1");
      RAJA::exclusive_scan_inplace<
        RAJA::hip_exec<block_size, true /*async*/> >(
          res,
          RAJA::make_span(counts+ibegin, iend+1-ibegin) );

      RP_CALI_SUBKERNEL_BEGIN("INDEXLIST_3LOOP_2");
      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend),
        [=] __device__ (Index_type i) {
        if (counts[i] != counts[i+1]) {
          list[counts[i]] = i;
        }
        if (i == iend-1) {
          *len = counts[i+1];
        }
      });
      RP_CALI_SUBKERNEL_END("INDEXLIST_3LOOP_2");

      res.wait();
      m_len = *len;

    }
    stopTimer();

    deallocData(DataSpace::HipPinnedCoarse, len);

    INDEXLIST_3LOOP_COUNTS_TEARDOWN(DataSpace::HipDevice);

  } else {
    getCout() << "\n  INDEXLIST_3LOOP : Unknown variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(INDEXLIST_3LOOP, Hip, Base_HIP, RAJA_HIP)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
