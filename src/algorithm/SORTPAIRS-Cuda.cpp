//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SORTPAIRS.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

#define SORTPAIRS_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, iend*run_reps); \
  allocAndInitCudaDeviceData(i, m_i, iend*run_reps);

#define SORTPAIRS_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x, x, iend*run_reps); \
  getCudaDeviceData(m_i, i, iend*run_reps); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(i);


void SORTPAIRS::runCudaVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SORTPAIRS_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    SORTPAIRS_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::sort_pairs< RAJA::cuda_exec<default_gpu_block_size, true /*async*/> >(RAJA_SORTPAIRS_ARGS);

    }
    stopTimer();

    SORTPAIRS_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  SORTPAIRS : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
