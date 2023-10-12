//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
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

#define INDEXLIST_3LOOP_DATA_SETUP_CUDA \
  Index_type* counts; \
  allocData(DataSpace::CudaDevice, counts, iend+1);

#define INDEXLIST_3LOOP_DATA_TEARDOWN_CUDA \
  deallocData(DataSpace::CudaDevice, counts);


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
void INDEXLIST_3LOOP::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  INDEXLIST_3LOOP_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    INDEXLIST_3LOOP_DATA_SETUP_CUDA;

    Index_type* len;
    allocData(DataSpace::CudaPinned, len, 1);

    cudaStream_t stream = res.get_stream();

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
    allocData(DataSpace::CudaDevice, temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;
      indexlist_conditional<block_size><<<grid_size, block_size, shmem, stream>>>(
          x, counts, iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                  temp_storage_bytes,
                                                  counts+ibegin,
                                                  counts+ibegin,
                                                  binary_op,
                                                  init_val,
                                                  scan_size,
                                                  stream));

      indexlist_make_list<block_size><<<grid_size, block_size, shmem, stream>>>(
          list, counts, len, iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk( cudaStreamSynchronize(stream) );
      m_len = *len;

    }
    stopTimer();

    deallocData(DataSpace::CudaDevice, temp_storage);
    deallocData(DataSpace::CudaPinned, len);

    INDEXLIST_3LOOP_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    INDEXLIST_3LOOP_DATA_SETUP_CUDA;

    Index_type* len;
    allocData(DataSpace::CudaPinned, len, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend),
        [=] __device__ (Index_type i) {
        counts[i] = (INDEXLIST_3LOOP_CONDITIONAL) ? 1 : 0;
      });

      RAJA::exclusive_scan_inplace< RAJA::cuda_exec<block_size, true /*async*/> >( res,
          RAJA::make_span(counts+ibegin, iend+1-ibegin));

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend),
        [=] __device__ (Index_type i) {
        if (counts[i] != counts[i+1]) {
          list[counts[i]] = i;
        }
        if (i == iend-1) {
          *len = counts[i+1];
        }
      });

      res.wait();
      m_len = *len;

    }
    stopTimer();

    deallocData(DataSpace::CudaPinned, len);

    INDEXLIST_3LOOP_DATA_TEARDOWN_CUDA;

  } else {
    getCout() << "\n  INDEXLIST_3LOOP : Unknown variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(INDEXLIST_3LOOP, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
