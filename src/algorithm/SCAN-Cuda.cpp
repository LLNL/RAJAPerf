//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SCAN.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define SCAN_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, iend); \
  allocAndInitCudaDeviceData(y, m_y, iend);

#define SCAN_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_y, y, iend); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y);


void SCAN::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  SCAN_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    SCAN_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::exclusive_scan< RAJA::cuda_exec<block_size, true /*async*/> >(RAJA_SCAN_ARGS);

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
