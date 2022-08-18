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

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

#define SCAN_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, iend); \
  allocAndInitHipDeviceData(y, m_y, iend);

#define SCAN_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_y, y, iend); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(y);


void SCAN::runHipVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SCAN_DATA_SETUP;

  if ( vid == Base_HIP ) {

    SCAN_DATA_SETUP_HIP;

    hipStream_t stream = 0;

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
    allocHipDeviceData(temp_storage, temp_storage_bytes);
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
    deallocHipDeviceData(temp_storage);

    SCAN_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    SCAN_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::exclusive_scan< RAJA::hip_exec<default_gpu_block_size, true /*async*/> >(RAJA_SCAN_ARGS);

    }
    stopTimer();

    SCAN_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  SCAN : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
