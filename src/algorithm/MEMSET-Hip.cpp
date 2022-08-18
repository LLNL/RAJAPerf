//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MEMSET.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

#define MEMSET_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, iend);

#define MEMSET_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_x, x, iend); \
  deallocHipDeviceData(x);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void memset(Real_ptr x, Real_type val,
                       Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if ( i < iend ) {
    MEMSET_BODY;
  }
}


void MEMSET::runHipVariantLibrary(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMSET_DATA_SETUP;

  if ( vid == Base_HIP ) {

    MEMSET_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk( hipMemsetAsync(MEMSET_STD_ARGS, 0) );

    }
    stopTimer();

    MEMSET_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    MEMSET_DATA_SETUP_HIP;

    camp::resources::Hip res = camp::resources::Hip::get_default();

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      res.memset(MEMSET_STD_ARGS);

    }
    stopTimer();

    MEMSET_DATA_TEARDOWN_HIP;

  } else {

    getCout() << "\n  MEMSET : Unknown Hip variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void MEMSET::runHipVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMSET_DATA_SETUP;

  if ( vid == Base_HIP ) {

    MEMSET_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (memset<block_size>),
          dim3(grid_size), dim3(block_size), 0, 0,
          x, val, iend );
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    MEMSET_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    MEMSET_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto memset_lambda = [=] __device__ (Index_type i) {
        MEMSET_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(memset_lambda)>),
          grid_size, block_size, 0, 0,
          ibegin, iend, memset_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    MEMSET_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    MEMSET_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          MEMSET_BODY;
      });

    }
    stopTimer();

    MEMSET_DATA_TEARDOWN_HIP;

  } else {

    getCout() << "\n  MEMSET : Unknown Hip variant id = " << vid << std::endl;

  }

}

void MEMSET::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if (vid == Base_HIP || vid == RAJA_HIP) {

    if (tune_idx == t) {

      runHipVariantLibrary(vid);

    }

    t += 1;

  }

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      if (tune_idx == t) {

        runHipVariantBlock<block_size>(vid);

      }

      t += 1;

    }

  });

}

void MEMSET::setHipTuningDefinitions(VariantID vid)
{
  if (vid == Base_HIP || vid == RAJA_HIP) {
    addVariantTuningName(vid, "library");
  }

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      addVariantTuningName(vid, "block_"+std::to_string(block_size));

    }

  });

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
