//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POINTER_CHASE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

__launch_bounds__(1)
__global__ void pointer_chase_cuda(Index_ptr next, Index_ptr result,
                                   Index_type start_idx,
                                   std::uint64_t num_steps)
{
  POINTER_CHASE_BODY;
}

void POINTER_CHASE::runCudaVariant(VariantID vid)
{
  setBlockSize(1);

  const Index_type run_reps = getRunReps();
  auto res{getCudaResource()};

  POINTER_CHASE_DATA_SETUP;

  if (vid == Base_CUDA) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps;
         RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POINTER_CHASE_1");
      RPlaunchCudaKernel((pointer_chase_cuda),
                         1, 1, 0, res.get_stream(),
                         next, result, start_idx, num_steps);
      RP_CALI_SUBKERNEL_END("POINTER_CHASE_1");

    }
    stopTimer();

  } else if (vid == Lambda_CUDA) {

    auto pointer_chase_lam = [=] __device__ (Index_type) {
      POINTER_CHASE_BODY;
    };

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps;
         RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POINTER_CHASE_1");
      RPlaunchCudaKernel((lambda_cuda_forall<1,
                                            decltype(pointer_chase_lam)>),
                         1, 1, 0, res.get_stream(),
                         Index_type(0), Index_type(1), pointer_chase_lam);
      RP_CALI_SUBKERNEL_END("POINTER_CHASE_1");

    }
    stopTimer();

  } else if (vid == RAJA_CUDA) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps;
         RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POINTER_CHASE_1");
      RAJA::forall<RAJA::cuda_exec<1, true>>(res,
        RAJA::RangeSegment(0, 1), [=] __device__ (Index_type) {
          POINTER_CHASE_BODY;
      });
      RP_CALI_SUBKERNEL_END("POINTER_CHASE_1");

    }
    stopTimer();

  } else {
    getCout() << "\n  POINTER_CHASE : Unknown Cuda variant id = " << vid
              << std::endl;
  }
}

void POINTER_CHASE::defineCudaVariantTunings()
{
  for (VariantID vid : {Base_CUDA, Lambda_CUDA, RAJA_CUDA}) {
    addVariantTuning<&POINTER_CHASE::runCudaVariant>(
        vid, getDefaultTuningName(), Index_type(1));
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
