//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SORT.hpp"

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


#define SORT_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, iend*run_reps);

#define SORT_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x, x, iend*run_reps); \
  deallocCudaDeviceData(x);


void SORT::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SORT_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    SORT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::sort< RAJA::cuda_exec<block_size, true /*async*/> >(RAJA_SORT_ARGS);

    }
    stopTimer();

    SORT_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  SORT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
