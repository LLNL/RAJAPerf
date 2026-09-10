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
    detail::hip::grid_scan_max_items_per_thread<Real_type, block_size>::value+1,
    integer::LessEqual<detail::hip::grid_scan_max_items_per_thread<Real_type, block_size>::value>>;


template < size_t block_size, size_t items_per_thread >
__launch_bounds__(block_size)
__global__ void scan_custom(Real_ptr x,
                            Real_ptr y,
                            Real_ptr block_counts,
                            Real_ptr grid_counts,
                            unsigned* block_readys,
                            Index_type iend)
{
  // It looks like blocks do not start running in order in hip, so a block
  // with a higher index can't wait on a block with a lower index without
  // deadlocking (have to replace with an atomicInc)
  const int block_id = blockIdx.x;

  Real_type vals[items_per_thread];

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    if (i < iend) {
      vals[ti] = x[i];
    } else {
      vals[ti] = 0;
    }
  }

  Real_type exclusives[items_per_thread];
  Real_type inclusives[items_per_thread];
  detail::hip::GridScan<Real_type, block_size, items_per_thread>::grid_scan(
      block_id, vals, exclusives, inclusives, block_counts, grid_counts, block_readys);

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

    RAJA::operators::plus<Real_type> binary_op;
    Real_type init_val = 0.0;

    int len = iend - ibegin;

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::exclusive_scan,
        d_temp_storage, temp_storage_bytes,
        x+ibegin,
        y+ibegin,
        init_val,
        len,
        binary_op,
        stream);
#elif defined(__CUDACC__)
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
        d_temp_storage, temp_storage_bytes,
        x+ibegin,
        y+ibegin,
        binary_op,
        init_val,
        len,
        stream);
#endif

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocData(DataSpace::HipDevice, temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("SCAN_1");
      // Run
#if defined(__HIPCC__)
      CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::exclusive_scan,
          d_temp_storage, temp_storage_bytes,
          x+ibegin,
          y+ibegin,
          init_val,
          len,
          binary_op,
          stream);
#elif defined(__CUDACC__)
      CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
          d_temp_storage, temp_storage_bytes,
          x+ibegin,
          y+ibegin,
          binary_op,
          init_val,
          len,
          stream);
#endif
      RP_CALI_SUBKERNEL_END("SCAN_1");

    }
    stopTimer();

    // Free temporary storage
    deallocData(DataSpace::HipDevice, temp_storage);

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("SCAN_1");
      RAJA::exclusive_scan< RAJA::hip_exec<0, true /*async*/> >(res, RAJA_SCAN_ARGS);
      RP_CALI_SUBKERNEL_END("SCAN_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  SCAN : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size, size_t items_per_thread >
void SCAN::runHipVariantCustom(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  SCAN_DATA_SETUP;

  if ( vid == Base_HIP ) {

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT((iend-ibegin), block_size*items_per_thread);
    const size_t shmem_size = 0;

    Real_ptr block_counts;
    allocData(DataSpace::HipDevice, block_counts, grid_size);
    Real_ptr grid_counts;
    allocData(DataSpace::HipDevice, grid_counts, grid_size);
    unsigned* block_readys;
    allocData(DataSpace::HipDevice, block_readys, grid_size);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("SCAN_1");
      CAMP_HIP_API_INVOKE_AND_CHECK( hipMemsetAsync,
          block_readys, 0, sizeof(unsigned)*grid_size, res.get_stream() );

      RPlaunchHipKernel( (scan_custom<block_size, items_per_thread>),
                         grid_size, block_size,
                         shmem_size, res.get_stream(),
                         x+ibegin, y+ibegin,
                         block_counts, grid_counts, block_readys,
                         iend-ibegin );
      RP_CALI_SUBKERNEL_END("SCAN_1");

    }
    stopTimer();

    deallocData(DataSpace::HipDevice, block_counts);
    deallocData(DataSpace::HipDevice, grid_counts);
    deallocData(DataSpace::HipDevice, block_readys);

  } else {
     getCout() << "\n  SCAN : Unknown Hip variant id = " << vid << std::endl;
  }
}


void SCAN::defineHipVariantTunings()
{

  for (VariantID vid : {Base_HIP, RAJA_HIP}) {

    addVariantTuning<&SCAN::runHipVariantLibrary>(
        vid, "rocprim");

    if ( vid == Base_HIP && run_params.getEnableCustomScan() ) {

      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {

          using hip_items_per_thread = hip_items_per_thread_type<block_size>;

          if (camp::size<hip_items_per_thread>::value == 0) {

            addVariantTuning<&SCAN::runHipVariantCustom<
                                 decltype(block_size)::value,
                                 detail::hip::grid_scan_default_items_per_thread<
                                     Real_type, block_size,
                                     RAJA_PERFSUITE_TUNING_HIP_ARCH>::value>>(
                vid, "block_"+std::to_string(block_size));

          }

          seq_for(hip_items_per_thread{}, [&](auto items_per_thread) {

            if (run_params.numValidItemsPerThread() == 0u ||
                run_params.validItemsPerThread(block_size)) {

              addVariantTuning<&SCAN::runHipVariantCustom<
                                   block_size,
                                   items_per_thread>>(
                  vid, "itemsPerThread<"+std::to_string(items_per_thread)+">_"
                       "block_"+std::to_string(block_size));

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
