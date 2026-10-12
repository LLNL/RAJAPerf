//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D_OFFSET.hpp"

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
__global__ void initview1d_offset(Real_ptr a,
                                  Real_type v,
                                  const Index_type ibegin,
                                  const Index_type iend)
{
  Index_type i = ibegin + blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    INIT_VIEW1D_OFFSET_BODY;
  }
}



template < size_t block_size >
void INIT_VIEW1D_OFFSET::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getActualProblemSize()+1;

  auto res{getCudaResource()};

  INIT_VIEW1D_OFFSET_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("INIT_VIEW1D_OFFSET_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
      constexpr size_t shmem = 0;
     
      RPlaunchCudaKernel( (initview1d_offset<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          a, v, 
                          ibegin, iend );
      RP_CALI_SUBKERNEL_END("INIT_VIEW1D_OFFSET_1");

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("INIT_VIEW1D_OFFSET_1");
      auto initview1d_offset_lambda = [=] __device__ (Index_type i) {
        INIT_VIEW1D_OFFSET_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (lambda_cuda_forall<block_size,
                                              decltype(initview1d_offset_lambda)>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          ibegin, iend, initview1d_offset_lambda ); 
      RP_CALI_SUBKERNEL_END("INIT_VIEW1D_OFFSET_1");

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    INIT_VIEW1D_OFFSET_VIEW_RAJA;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("INIT_VIEW1D_OFFSET_1");
      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        INIT_VIEW1D_OFFSET_BODY_RAJA;
      });
      RP_CALI_SUBKERNEL_END("INIT_VIEW1D_OFFSET_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  INIT_VIEW1D_OFFSET : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(INIT_VIEW1D_OFFSET, Cuda, Base_CUDA, Lambda_CUDA, RAJA_CUDA)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
