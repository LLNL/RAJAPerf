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

template < size_t block_size, size_t items_per_thread >
__launch_bounds__(block_size)
__global__ void scan(Real_ptr x,
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
  detail::cuda::grid_scan<block_size, items_per_thread>(
      block_id, vals, exclusives, inclusives, block_counts, grid_counts, block_readys);

  for (size_t ti = 0; ti < items_per_thread; ++ti) {
    Index_type i = block_id * block_size * items_per_thread + ti * block_size + threadIdx.x;
    if (i < iend) {
      y[i] = exclusives[ti];
    }
  }
}


void SCAN::runCudaVariantCub(VariantID vid)
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
    cudaErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                temp_storage_bytes,
                                                x+ibegin,
                                                y+ibegin,
                                                binary_op,
                                                init_val,
                                                len,
                                                stream));

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocData(DataSpace::CudaDevice, temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Run
      cudaErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                  temp_storage_bytes,
                                                  x+ibegin,
                                                  y+ibegin,
                                                  binary_op,
                                                  init_val,
                                                  len,
                                                  stream));

    }
    stopTimer();

    // Free temporary storage
    deallocData(DataSpace::CudaDevice, temp_storage);

  } else {
     getCout() << "\n  SCAN : Unknown Cuda variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void SCAN::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  SCAN_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    const size_t grid_size = RAJA_DIVIDE_CEILING_INT((iend-ibegin), block_size*detail::cuda::grid_scan_items_per_thread);
    const size_t shmem_size = 0;

    Real_ptr block_counts;
    allocData(DataSpace::CudaDevice, block_counts, grid_size);
    Real_ptr grid_counts;
    allocData(DataSpace::CudaDevice, grid_counts, grid_size);
    unsigned* block_readys;
    allocData(DataSpace::CudaDevice, block_readys, grid_size);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      cudaErrchk( cudaMemsetAsync(block_readys, 0, sizeof(unsigned)*grid_size,
                                  res.get_stream()) );
      RPlaunchCudaKernel( (scan<block_size, detail::cuda::grid_scan_items_per_thread>),
                          grid_size, block_size,
                          shmem_size, res.get_stream(),
                          x+ibegin, y+ibegin,
                          block_counts, grid_counts, block_readys,
                          iend-ibegin );

    }
    stopTimer();

    deallocData(DataSpace::CudaDevice, block_counts);
    deallocData(DataSpace::CudaDevice, grid_counts);
    deallocData(DataSpace::CudaDevice, block_readys);

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::exclusive_scan< RAJA::cuda_exec<default_gpu_block_size, true /*async*/> >(res, RAJA_SCAN_ARGS);

    }
    stopTimer();

  } else {
     getCout() << "\n  SCAN : Unknown Cuda variant id = " << vid << std::endl;
  }
}


void SCAN::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_CUDA ) {

    if (tune_idx == t) {

      runCudaVariantCub(vid);

    }

    t += 1;

  }

  if ( vid == Base_CUDA || vid == RAJA_CUDA ) {

    if (tune_idx == t) {

      runCudaVariantImpl<default_gpu_block_size>(vid);

    }

    t += 1;

  } else {

    getCout() << "\n  SCAN : Unknown Cuda variant id = " << vid << std::endl;

  }
}

void SCAN::setCudaTuningDefinitions(VariantID vid)
{
  if ( vid == Base_CUDA ) {

    addVariantTuningName(vid, "cub");

  }

  if ( vid == Base_CUDA || vid == RAJA_CUDA ) {

    addVariantTuningName(vid, "default");

  }
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
