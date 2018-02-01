//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_DIFF.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define FIRST_DIFF_DATA_SETUP_CUDA \
  Real_ptr x; \
  Real_ptr y; \
\
  allocAndInitCudaDeviceData(x, m_x, m_array_length); \
  allocAndInitCudaDeviceData(y, m_y, m_array_length);

#define FIRST_DIFF_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x, x, m_array_length); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y);

__global__ void first_diff(Real_ptr x, Real_ptr y,
                           Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     FIRST_DIFF_BODY; 
   }
}


void FIRST_DIFF::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_CUDA ) {

    FIRST_DIFF_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       first_diff<<<grid_size, block_size>>>( x, y,
                                              iend ); 

    }
    stopTimer();

    FIRST_DIFF_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    FIRST_DIFF_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         FIRST_DIFF_BODY;
       });

    }
    stopTimer();

    FIRST_DIFF_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  FIRST_DIFF : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
