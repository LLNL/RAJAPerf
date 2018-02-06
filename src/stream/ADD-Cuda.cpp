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

#include "ADD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define ADD_DATA_SETUP_CUDA \
  Real_ptr a; \
  Real_ptr b; \
  Real_ptr c; \
\
  allocAndInitCudaDeviceData(a, m_a, iend); \
  allocAndInitCudaDeviceData(b, m_b, iend); \
  allocAndInitCudaDeviceData(c, m_c, iend);

#define ADD_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_c, c, iend); \
  deallocCudaDeviceData(a); \
  deallocCudaDeviceData(b); \
  deallocCudaDeviceData(c);

__global__ void add(Real_ptr c, Real_ptr a, Real_ptr b,
                     Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ADD_BODY;
   }
}


void ADD::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_CUDA ) {

    ADD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      add<<<grid_size, block_size>>>( c, a, b,
                                      iend );

    }
    stopTimer();

    ADD_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    ADD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        ADD_BODY;
      });

    }
    stopTimer();

    ADD_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  ADD : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA

