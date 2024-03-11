//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"
#include "common/CudaGridScan.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t block_size >
using cuda_items_per_thread_type = integer::make_gpu_items_per_thread_list_type<
    detail::cuda::grid_scan_default_items_per_thread<INDEXLIST::Idx_type, block_size, RAJA_PERFSUITE_TUNING_CUDA_ARCH>::value,
    integer::LessEqual<detail::cuda::grid_scan_max_items_per_thread<INDEXLIST::Idx_type, block_size>::value>>;


template < size_t block_size, size_t items_per_thread >
using GridScan = detail::cuda::GridScan<INDEXLIST::Idx_type, block_size, items_per_thread>;

template < size_t block_size, size_t items_per_thread >
using GridScanDeviceStorage = typename GridScan<block_size, items_per_thread>::DeviceStorage;

template < size_t block_size, size_t items_per_thread >
__launch_bounds__(block_size)
__global__ void indexlist(Real_ptr x,
                          INDEXLIST::Idx_ptr list,
                          GridScanDeviceStorage<block_size, items_per_thread> device_storage,
                          INDEXLIST::Idx_type* len,
                          Index_type iend)
{
  // blocks do start running in order in cuda, so a block with a higher
  // index can wait on a block with a lower index without deadlocking
  // (replace with an atomicInc if this changes)
  const int block_id = blockIdx.x;

  INDEXLIST::Idx_type vals[items_per_thread];

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    INDEXLIST::Idx_type val = 0;
    if (i < iend) {
      if (INDEXLIST_CONDITIONAL) {
        val = 1;
      }
    }
    vals[ti] = val;
  }

  INDEXLIST::Idx_type exclusives[items_per_thread];
  INDEXLIST::Idx_type inclusives[items_per_thread];
  GridScan<block_size, items_per_thread>::grid_scan(
      block_id, vals, exclusives, inclusives, device_storage);

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    INDEXLIST::Idx_type exclusive = exclusives[ti];
    INDEXLIST::Idx_type inclusive = inclusives[ti];
    if (i < iend) {
      if (exclusive != inclusive) {
        list[exclusive] = i;
      }
      if (i == iend-1) {
        *len = inclusive;
      }
    }
  }
}


template < size_t block_size, size_t items_per_thread >
void INDEXLIST::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  INDEXLIST_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    using GridScan = GridScan<block_size, items_per_thread>;
    using GridScanDeviceStorage = typename GridScan::DeviceStorage;

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT((iend-ibegin), block_size*items_per_thread);
    const size_t shmem_size = 0;

    Idx_type* len;
    allocData(DataSpace::CudaPinned, len, 1);

    GridScanDeviceStorage device_storage;
    void* storage_to_zero = nullptr;
    size_t storage_size = 0;
    device_storage.allocate(storage_to_zero, storage_size,
        grid_size, [&](auto& ptr, size_t size) {
          allocData(DataSpace::CudaDevice, ptr, size);
        });

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      if (storage_to_zero) {
        cudaErrchk( cudaMemsetAsync(storage_to_zero, 0, storage_size, res.get_stream()) );
      }

      RPlaunchCudaKernel( (indexlist<block_size, items_per_thread>),
                          grid_size, block_size,
                          shmem_size, res.get_stream(),
                          x+ibegin, list+ibegin,
                          device_storage,
                          len, iend-ibegin );

      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );
      m_len = *len;

    }
    stopTimer();

    deallocData(DataSpace::CudaPinned, len);
    device_storage.deallocate([&](auto& ptr) {
          deallocData(DataSpace::CudaDevice, ptr);
        });

  } else {
    getCout() << "\n  INDEXLIST : Unknown variant id = " << vid << std::endl;
  }
}


void INDEXLIST::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(cuda_items_per_thread_type<block_size>{}, [&](auto items_per_thread) {

          if (run_params.numValidItemsPerThread() == 0u ||
              run_params.validItemsPerThread(block_size)) {

            if (tune_idx == t) {

              runCudaVariantImpl<decltype(block_size)::value, items_per_thread>(vid);

            }

            t += 1;

          }

        });

      }

    });

  } else {

    getCout() << "\n  INDEXLIST : Unknown Cuda variant id = " << vid << std::endl;

  }
}

void INDEXLIST::setCudaTuningDefinitions(VariantID vid)
{
  if ( vid == Base_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(cuda_items_per_thread_type<block_size>{}, [&](auto items_per_thread) {

          if (run_params.numValidItemsPerThread() == 0u ||
              run_params.validItemsPerThread(block_size)) {

            addVariantTuningName(vid, "block_"+std::to_string(block_size)+
                                      "_itemsPerThread_"+std::to_string(items_per_thread));

          }

        });

      }

    });

  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
