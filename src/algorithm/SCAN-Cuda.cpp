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

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

#define SCAN_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, iend); \
  allocAndInitCudaDeviceData(y, m_y, iend);

#define SCAN_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_y, y, iend); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y);


void SCAN::runCudaVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SCAN_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    SCAN_DATA_SETUP_CUDA;

    cudaStream_t stream = 0;

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
    allocCudaDeviceData(temp_storage, temp_storage_bytes);
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
    deallocCudaDeviceData(temp_storage);

    SCAN_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    SCAN_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::exclusive_scan< RAJA::cuda_exec<default_gpu_block_size, true /*async*/> >(RAJA_SCAN_ARGS);

    }
    stopTimer();

    SCAN_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  SCAN : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
