//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// _add_run_cuda_start
#include "ADD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void add(Real_ptr c, Real_ptr a, Real_ptr b,
                    Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    ADD_BODY;
  }
}


template < size_t block_size >
void ADD::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  ADD_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("ADD_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (add<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          c, a, b, iend );
      RP_CALI_SUBKERNEL_END("ADD_1");

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("ADD_1");
      auto add_lambda = [=] __device__ (Index_type i) {
        ADD_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (lambda_cuda_forall<block_size, 
                                              decltype(add_lambda)>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          ibegin, iend, add_lambda );
      RP_CALI_SUBKERNEL_END("ADD_1");

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("ADD_1");
      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        ADD_BODY;
      });
      RP_CALI_SUBKERNEL_END("ADD_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  ADD : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(ADD, Cuda, Base_CUDA, Lambda_CUDA, RAJA_CUDA)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
// _add_run_cuda_end

