//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST_3LOOP.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define INDEXLIST_3LOOP_DATA_SETUP_CUDA \
  Index_type* counts; \
  allocCudaDeviceData(counts, getRunSize()+1); \
  allocAndInitCudaDeviceData(x, m_x, iend); \
  allocAndInitCudaDeviceData(list, m_list, iend);

#define INDEXLIST_3LOOP_DATA_TEARDOWN_CUDA \
  deallocCudaDeviceData(counts); \
  getCudaDeviceData(m_list, list, iend); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(list);


__global__ void indexlist_conditional(Real_ptr x,
                                      Int_ptr list,
                                      Index_type* counts,
                                      Index_type iend)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < iend) {
    counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
  }
}

__global__ void indexlist_make_list(Int_ptr list,
                                    Index_type* counts,
                                    Index_type* len,
                                    Index_type iend)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < iend) {
    if (counts[i] != counts[i+1]) {
      list[counts[i]] = i;
    }
    if (i == iend-1) {
      *len = counts[i+1];
    }
  }
}


void INDEXLIST_3LOOP::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  INDEXLIST_3LOOP_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    INDEXLIST_3LOOP_DATA_SETUP_CUDA;

    Index_type* len;
    allocCudaPinnedData(len, 1);

    cudaStream_t stream = RAJA::resources::Cuda::get_default().get_stream();

    RAJA::operators::plus<Index_type> binary_op;
    Index_type init_val = 0;
    int scan_size = iend+1 - ibegin;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cudaErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                temp_storage_bytes,
                                                counts+ibegin,
                                                counts+ibegin,
                                                binary_op,
                                                init_val,
                                                scan_size,
                                                stream));

    unsigned char* temp_storage;
    allocCudaDeviceData(temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      indexlist_conditional<<<grid_size, block_size, 0, stream>>>(
          x, list, counts, iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                  temp_storage_bytes,
                                                  counts+ibegin,
                                                  counts+ibegin,
                                                  binary_op,
                                                  init_val,
                                                  scan_size,
                                                  stream));

      indexlist_make_list<<<grid_size, block_size, 0, stream>>>(
          list, counts, len, iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk( cudaStreamSynchronize(stream) );
      m_len = *len;

    }
    stopTimer();

    deallocCudaDeviceData(temp_storage);
    deallocCudaPinnedData(len);

    INDEXLIST_3LOOP_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    INDEXLIST_3LOOP_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Index_type> len(0);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend),
        [=] __device__ (Index_type i) {
        counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
      });

      RAJA::exclusive_scan_inplace< RAJA::cuda_exec<block_size, true /*async*/> >(
          RAJA::make_span(counts+ibegin, iend+1-ibegin));

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
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

    INDEXLIST_3LOOP_DATA_TEARDOWN_CUDA;

  } else {
    std::cout << "\n  INDEXLIST_3LOOP : Unknown variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
