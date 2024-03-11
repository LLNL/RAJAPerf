//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SCAN.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#if defined(__HIPCC__)
#define ROCPRIM_HIP_API 1
#include "rocprim/device/device_scan.hpp"
#elif defined(__CUDACC__)
#include "cub/device/device_scan.cuh"
#include "cub/util_allocator.cuh"
#endif

#include "common/HipDataUtils.hpp"
#include "common/HipGridScan.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

template < size_t block_size >
using hip_items_per_thread_type = integer::make_gpu_items_per_thread_list_type<
    detail::hip::grid_scan_default_items_per_thread<SCAN::Data_type, block_size, RAJA_PERFSUITE_TUNING_HIP_ARCH>::value,
    integer::LessEqual<detail::hip::grid_scan_max_items_per_thread<SCAN::Data_type, block_size>::value>>;

template < size_t block_size, size_t items_per_thread >
using GridScan = detail::hip::GridScan<SCAN::Data_type, block_size, items_per_thread>;

template < size_t block_size, size_t items_per_thread >
using GridScanDeviceStorage = typename GridScan<block_size, items_per_thread>::DeviceStorage;

template < size_t block_size, size_t items_per_thread >
__launch_bounds__(block_size)
__global__ void scan(SCAN::Data_ptr x,
                     SCAN::Data_ptr y,
                     GridScanDeviceStorage<block_size, items_per_thread> device_storage,
                     Index_type iend)
{
  // It looks like blocks do not start running in order in hip, so a block
  // with a higher index can't wait on a block with a lower index without
  // deadlocking (have to replace with an atomicInc)
  const int block_id = blockIdx.x;

  SCAN::Data_type vals[items_per_thread];

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    if (i < iend) {
      vals[ti] = x[i];
    } else {
      vals[ti] = 0;
    }
  }

  SCAN::Data_type exclusives[items_per_thread];
  SCAN::Data_type inclusives[items_per_thread];
  GridScan<block_size, items_per_thread>::grid_scan(
      block_id, vals, exclusives, inclusives, device_storage);

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    if (i < iend) {
      y[i] = exclusives[ti];
    }
  }
}


void SCAN::runHipVariantLibrary(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  SCAN_DATA_SETUP;

  if ( vid == Base_HIP ) {

    hipStream_t stream = res.get_stream();

    RAJA::operators::plus<Data_type> binary_op;
    Data_type init_val = 0;

    int len = iend - ibegin;

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
    hipErrchk(::rocprim::exclusive_scan(d_temp_storage,
                                        temp_storage_bytes,
                                        x+ibegin,
                                        y+ibegin,
                                        init_val,
                                        len,
                                        binary_op,
                                        stream));
#elif defined(__CUDACC__)
    hipErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                               temp_storage_bytes,
                                               x+ibegin,
                                               y+ibegin,
                                               binary_op,
                                               init_val,
                                               len,
                                               stream));
#endif

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocData(DataSpace::HipDevice, temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Run
#if defined(__HIPCC__)
      hipErrchk(::rocprim::exclusive_scan(d_temp_storage,
                                          temp_storage_bytes,
                                          x+ibegin,
                                          y+ibegin,
                                          init_val,
                                          len,
                                          binary_op,
                                          stream));
#elif defined(__CUDACC__)
      hipErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                 temp_storage_bytes,
                                                 x+ibegin,
                                                 y+ibegin,
                                                 binary_op,
                                                 init_val,
                                                 len,
                                                 stream));
#endif

    }
    stopTimer();

    // Free temporary storage
    deallocData(DataSpace::HipDevice, temp_storage);

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::exclusive_scan< RAJA::hip_exec<0, true /*async*/> >(res, RAJA_SCAN_ARGS);

    }
    stopTimer();

  } else {
     getCout() << "\n  SCAN : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size, size_t items_per_thread >
void SCAN::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  SCAN_DATA_SETUP;

  if ( vid == Base_HIP ) {

    using GridScan = GridScan<block_size, items_per_thread>;
    using GridScanDeviceStorage = typename GridScan::DeviceStorage;

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT((iend-ibegin), block_size*items_per_thread);
    const size_t shmem_size = 0;

    GridScanDeviceStorage device_storage;
    void* storage_to_zero = nullptr;
    size_t storage_size = 0;
    device_storage.allocate(storage_to_zero, storage_size,
        grid_size, [&](auto& ptr, size_t size) {
          allocData(DataSpace::HipDevice, ptr, size);
        });

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      if (storage_to_zero) {
        hipErrchk( hipMemsetAsync(storage_to_zero, 0, storage_size, res.get_stream()) );
      }

      RPlaunchHipKernel( (scan<block_size, items_per_thread>),
                         grid_size, block_size,
                         shmem_size, res.get_stream(),
                         x+ibegin, y+ibegin,
                         device_storage,
                         iend-ibegin );

    }
    stopTimer();

    device_storage.deallocate([&](auto& ptr) {
          deallocData(DataSpace::HipDevice, ptr);
        });

  } else {
     getCout() << "\n  SCAN : Unknown Hip variant id = " << vid << std::endl;
  }
}


void SCAN::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_HIP || vid == RAJA_HIP ) {

    if (tune_idx == t) {

      runHipVariantLibrary(vid);

    }

    t += 1;

    if ( vid == Base_HIP ) {

      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {

          seq_for(hip_items_per_thread_type<block_size>{}, [&](auto items_per_thread) {

            if (run_params.numValidItemsPerThread() == 0u ||
                run_params.validItemsPerThread(block_size)) {

              if (tune_idx == t) {

                runHipVariantImpl<block_size, items_per_thread>(vid);

              }

              t += 1;

            }

          });

        }

      });
    }

  } else {

    getCout() << "\n  SCAN : Unknown Hip variant id = " << vid << std::endl;

  }
}

void SCAN::setHipTuningDefinitions(VariantID vid)
{
  if ( vid == Base_HIP || vid == RAJA_HIP ) {

    addVariantTuningName(vid, "rocprim");

    if ( vid == Base_HIP ) {

      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {

          seq_for(hip_items_per_thread_type<block_size>{}, [&](auto items_per_thread) {

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
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
