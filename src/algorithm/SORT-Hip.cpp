//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SORT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{
#define SORT_ALLOC_HIP_DATA \
  allocHipDeviceData(x, iend*run_reps); \

#define SORT_INIT_HIP_DATA \
  initHipDeviceData(x, m_x, iend*run_reps); \

#define SORT_GET_HIP_DEVICE_DATA \
  getHipDeviceData(m_x, x, iend*run_reps);

#define SORT_DEALLOC_HIP_DATA \
  deallocHipDeviceData(x);


void SORT::runHipVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SORT_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    SORT_ALLOC_HIP_DATA;
    SORT_INIT_HIP_DATA;    

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::sort< RAJA::hip_exec<default_gpu_block_size, true /*async*/> >(RAJA_SORT_ARGS);

    }
    stopTimer();

    SORT_GET_HIP_DEVICE_DATA;
    SORT_DEALLOC_HIP_DATA;

  } else {
     getCout() << "\n  SORT : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
