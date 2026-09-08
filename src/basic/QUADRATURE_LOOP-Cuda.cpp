//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "QUADRATURE_LOOP.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void quadrature_loop(Real_ptr values, Real_ptr weights,
                                Real_ptr output,
                                Index_type num_zones)
{
  Index_type idx = blockIdx.x * block_size + threadIdx.x;
  if (idx < 27 * num_zones) {
    Index_type zone = idx / 27;
    Index_type q = idx - 27 * zone;
    QUADRATURE_LOOP_BODY;
  }
}



template < size_t block_size >
void QUADRATURE_LOOP::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  QUADRATURE_LOOP_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("QUADRATURE_LOOP_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (quadrature_loop<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          values, weights, output,
                          num_zones );
      RP_CALI_SUBKERNEL_END("QUADRATURE_LOOP_1");

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("QUADRATURE_LOOP_1");
      auto quadratureloop_lam = [=] __device__ (Index_type idx) {
        Index_type zone = idx / 27;
        Index_type q = idx - 27 * zone;
        QUADRATURE_LOOP_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (lambda_cuda_forall<block_size,
                                              decltype(quadratureloop_lam)>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          ibegin, iend, quadratureloop_lam );
      RP_CALI_SUBKERNEL_END("QUADRATURE_LOOP_1");

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    using FORNEST_POL =
      RAJA::fornest_collapsed_policy<RAJA::cuda_exec<block_size, true /*async*/>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("QUADRATURE_LOOP_1");
      RAJA::fornest(res, FORNEST_POL {},
                    RAJA::range(num_zones), RAJA::range(27),
                    [=] __device__ (Index_type zone, Index_type q) {
        QUADRATURE_LOOP_BODY;
      });
      RP_CALI_SUBKERNEL_END("QUADRATURE_LOOP_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  QUADRATURE_LOOP : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(QUADRATURE_LOOP, Cuda, Base_CUDA, Lambda_CUDA, RAJA_CUDA)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
