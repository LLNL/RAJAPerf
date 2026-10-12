//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SCAN.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "cub/device/device_scan.cuh"
#include "cub/util_allocator.cuh"

#include "common/CudaDataUtils.hpp"
#include "common/CudaGridScan.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

template < size_t block_size >
using cuda_items_per_thread_type = integer::make_gpu_items_per_thread_list_type<
    detail::cuda::grid_scan_max_items_per_thread<Real_type, block_size>::value+1,
    integer::LessEqual<detail::cuda::grid_scan_max_items_per_thread<Real_type, block_size>::value>>;


template < size_t block_size, size_t items_per_thread >
__launch_bounds__(block_size)
__global__ void scan_custom(Real_ptr x,
                            Real_ptr y,
                            Real_ptr block_counts,
                            Real_ptr grid_counts,
                            unsigned* block_readys,
                            Index_type iend)
{
  // blocks do start running in order in cuda, so a block with a higher
  // index can wait on a block with a lower index without deadlocking
  // (replace with an atomicInc if this changes)
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
  detail::cuda::GridScan<Real_type, block_size, items_per_thread>::grid_scan(
      block_id, vals, exclusives, inclusives, block_counts, grid_counts, block_readys);

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    if (i < iend) {
      y[i] = exclusives[ti];
    }
  }
}


void SCAN::runCudaVariantLibrary(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  SCAN_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    cudaStream_t stream = res.get_stream();

    RAJA::operators::plus<Real_type> binary_op;
    Real_type init_val = 0.0;

    int len = iend - ibegin;

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
        d_temp_storage, temp_storage_bytes,
        x+ibegin,
        y+ibegin,
        binary_op,
        init_val,
        len,
        stream);

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocData(DataSpace::CudaDevice, temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("SCAN_1");
      // Run
      CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceScan::ExclusiveScan,
          d_temp_storage, temp_storage_bytes,
          x+ibegin,
          y+ibegin,
          binary_op,
          init_val,
          len,
          stream);
      RP_CALI_SUBKERNEL_END("SCAN_1");

    }
    stopTimer();

    // Free temporary storage
    deallocData(DataSpace::CudaDevice, temp_storage);

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("SCAN_1");
      RAJA::exclusive_scan< RAJA::cuda_exec<0, true /*async*/> >(res, RAJA_SCAN_ARGS);
      RP_CALI_SUBKERNEL_END("SCAN_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  SCAN : Unknown Cuda variant id = " << vid << std::endl;
  }
}

template < size_t block_size, size_t items_per_thread >
void SCAN::runCudaVariantCustom(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  SCAN_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT((iend-ibegin), block_size*items_per_thread);
    const size_t shmem_size = 0;

    Real_ptr block_counts;
    allocData(DataSpace::CudaDevice, block_counts, grid_size);
    Real_ptr grid_counts;
    allocData(DataSpace::CudaDevice, grid_counts, grid_size);
    unsigned* block_readys;
    allocData(DataSpace::CudaDevice, block_readys, grid_size);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("SCAN_1");
      CAMP_CUDA_API_INVOKE_AND_CHECK( cudaMemsetAsync,
          block_readys, 0, sizeof(unsigned)*grid_size, res.get_stream() );
      RPlaunchCudaKernel( (scan_custom<block_size, items_per_thread>),
                          grid_size, block_size,
                          shmem_size, res.get_stream(),
                          x+ibegin, y+ibegin,
                          block_counts, grid_counts, block_readys,
                          iend-ibegin );
      RP_CALI_SUBKERNEL_END("SCAN_1");

    }
    stopTimer();

    deallocData(DataSpace::CudaDevice, block_counts);
    deallocData(DataSpace::CudaDevice, grid_counts);
    deallocData(DataSpace::CudaDevice, block_readys);

  } else {
     getCout() << "\n  SCAN : Unknown Cuda variant id = " << vid << std::endl;
  }
}


void SCAN::defineCudaVariantTunings()
{

  for (VariantID vid : {Base_CUDA, RAJA_CUDA}) {

    addVariantTuning<&SCAN::runCudaVariantLibrary>(
        vid, "cub");

    if ( vid == Base_CUDA && run_params.getEnableCustomScan() ) {

      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {

          using cuda_items_per_thread = cuda_items_per_thread_type<block_size>;

          if (camp::size<cuda_items_per_thread>::value == 0) {

            addVariantTuning<&SCAN::runCudaVariantCustom<
                                 decltype(block_size)::value,
                                 detail::cuda::grid_scan_default_items_per_thread<
                                     Real_type, block_size,
                                     RAJA_PERFSUITE_TUNING_CUDA_ARCH>::value>>(
                vid, "block_"+std::to_string(block_size));

          }

          seq_for(cuda_items_per_thread{}, [&](auto items_per_thread) {

            if (run_params.numValidItemsPerThread() == 0u ||
                run_params.validItemsPerThread(block_size)) {

              addVariantTuning<&SCAN::runCudaVariantCustom<
                                   decltype(block_size)::value,
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

#endif  // RAJA_ENABLE_CUDA
