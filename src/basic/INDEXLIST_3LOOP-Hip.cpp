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

#define INDEXLIST_3LOOP_DATA_SETUP_HIP \
  Index_type* counts; \
  allocHipDeviceData(counts, iend+1); \
  allocAndInitHipDeviceData(x, m_x, iend); \
  allocAndInitHipDeviceData(list, m_list, iend);

#define INDEXLIST_3LOOP_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(counts); \
  getHipDeviceData(m_list, list, iend); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(list);


template < size_t block_size >
__launch_bounds__(block_size)
__global__ void indexlist_conditional(Real_ptr x,
                                      Index_type* counts,
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
                                    Index_type* counts,
                                    Index_type* len,
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
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INDEXLIST_3LOOP_DATA_SETUP;

  if ( vid == Base_HIP ) {

    INDEXLIST_3LOOP_DATA_SETUP_HIP;

    Index_type* len;
    allocHipPinnedData(len, 1);

    hipStream_t stream = RAJA::resources::Hip::get_default().get_stream();

    RAJA::operators::plus<Index_type> binary_op;
    Index_type init_val = 0;
    int scan_size = iend+1 - ibegin;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
    hipErrchk(::rocprim::exclusive_scan(d_temp_storage,
                                        temp_storage_bytes,
                                        counts+ibegin,
                                        counts+ibegin,
                                        init_val,
                                        scan_size,
                                        binary_op,
                                        stream));
#elif defined(__CUDACC__)
    hipErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                               temp_storage_bytes,
                                               counts+ibegin,
                                               counts+ibegin,
                                               binary_op,
                                               init_val,
                                               scan_size,
                                               stream));
#endif

    unsigned char* temp_storage;
    allocHipDeviceData(temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((indexlist_conditional<block_size>), grid_size, block_size, 0, stream,
          x, counts, iend );
      hipErrchk( hipGetLastError() );

#if defined(__HIPCC__)
      hipErrchk(::rocprim::exclusive_scan(d_temp_storage,
                                          temp_storage_bytes,
                                          counts+ibegin,
                                          counts+ibegin,
                                          init_val,
                                          scan_size,
                                          binary_op,
                                          stream));
#elif defined(__CUDACC__)
      hipErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                 temp_storage_bytes,
                                                 counts+ibegin,
                                                 counts+ibegin,
                                                 binary_op,
                                                 init_val,
                                                 scan_size,
                                                 stream));
#endif

      hipLaunchKernelGGL((indexlist_make_list<block_size>), grid_size, block_size, 0, stream,
          list, counts, len, iend );
      hipErrchk( hipGetLastError() );

      hipErrchk( hipStreamSynchronize(stream) );
      m_len = *len;

    }
    stopTimer();

    deallocHipDeviceData(temp_storage);
    deallocHipPinnedData(len);

    INDEXLIST_3LOOP_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    INDEXLIST_3LOOP_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce, Index_type> len(0);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend),
        [=] __device__ (Index_type i) {
        counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
      });

      RAJA::exclusive_scan_inplace< RAJA::hip_exec<block_size, true /*async*/> >(
          RAJA::make_span(counts+ibegin, iend+1-ibegin));

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend),
        [=] __device__ (Index_type i) {
        if (counts[i] != counts[i+1]) {
          list[counts[i]] = i;
          len += 1;
        }
      });

      m_len = len.get();

    }
    stopTimer();

    INDEXLIST_3LOOP_DATA_TEARDOWN_HIP;

  } else {
    getCout() << "\n  INDEXLIST_3LOOP : Unknown variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(INDEXLIST_3LOOP, Hip)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
