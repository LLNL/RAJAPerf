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

template < size_t block_size, size_t items_per_thread >
__launch_bounds__(block_size)
__global__ void scan(Real_ptr x,
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
  detail::hip::grid_scan<block_size, items_per_thread>(
      block_id, vals, exclusives, inclusives, block_counts, grid_counts, block_readys);

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    if (i < iend) {
      y[i] = exclusives[ti];
    }
  }
}


void SCAN::runHipVariantRocprim(VariantID vid)
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

  } else {
     getCout() << "\n  SCAN : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void SCAN::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  SCAN_DATA_SETUP;

  if ( vid == Base_HIP ) {

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT((iend-ibegin), block_size*detail::hip::grid_scan_items_per_thread);
    const size_t shmem_size = 0;

    Real_ptr block_counts;
    allocData(DataSpace::HipDevice, block_counts, grid_size);
    Real_ptr grid_counts;
    allocData(DataSpace::HipDevice, grid_counts, grid_size);
    unsigned* block_readys;
    allocData(DataSpace::HipDevice, block_readys, grid_size);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk( hipMemsetAsync(block_readys, 0, sizeof(unsigned)*grid_size,
                                res.get_stream()) );

      RPlaunchHipKernel( (scan<block_size, detail::hip::grid_scan_items_per_thread>),
                         grid_size, block_size,
                         shmem_size, res.get_stream(),
                         x+ibegin, y+ibegin,
                         block_counts, grid_counts, block_readys,
                         iend-ibegin );

    }
    stopTimer();

    deallocData(DataSpace::HipDevice, block_counts);
    deallocData(DataSpace::HipDevice, grid_counts);
    deallocData(DataSpace::HipDevice, block_readys);

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::exclusive_scan< RAJA::hip_exec<block_size, true /*async*/> >(res, RAJA_SCAN_ARGS);

    }
    stopTimer();

  } else {
     getCout() << "\n  SCAN : Unknown Hip variant id = " << vid << std::endl;
  }
}


void SCAN::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_HIP ) {

    if (tune_idx == t) {

      runHipVariantRocprim(vid);

    }

    t += 1;

  }

  if ( vid == Base_HIP || vid == RAJA_HIP ) {

    if (tune_idx == t) {

      runHipVariantImpl<default_gpu_block_size>(vid);

    }

    t += 1;

  } else {

    getCout() << "\n  SCAN : Unknown Hip variant id = " << vid << std::endl;

  }
}

void SCAN::setHipTuningDefinitions(VariantID vid)
{
  if ( vid == Base_HIP ) {

    addVariantTuningName(vid, "rocprim");

  }

  if ( vid == Base_HIP || vid == RAJA_HIP ) {

    addVariantTuningName(vid, "default");

  }
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
