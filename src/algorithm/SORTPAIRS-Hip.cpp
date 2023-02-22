//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SORTPAIRS.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

#define SORTPAIRS_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, iend*run_reps); \
  allocAndInitHipDeviceData(i, m_i, iend*run_reps);

#define SORTPAIRS_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_x, x, iend*run_reps); \
  getHipDeviceData(m_i, i, iend*run_reps); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(i);


void SORTPAIRS::runHipVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SORTPAIRS_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    SORTPAIRS_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::sort_pairs< RAJA::hip_exec<default_gpu_block_size, true /*async*/> >(RAJA_SORTPAIRS_ARGS);

    }
    stopTimer();

    SORTPAIRS_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  SORTPAIRS : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
