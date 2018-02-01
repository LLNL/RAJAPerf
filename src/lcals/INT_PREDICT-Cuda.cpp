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

#include "INT_PREDICT.hpp"

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


#define INT_PREDICT_DATA_SETUP_CUDA \
  Real_ptr px; \
  Real_type dm22 = m_dm22; \
  Real_type dm23 = m_dm23; \
  Real_type dm24 = m_dm24; \
  Real_type dm25 = m_dm25; \
  Real_type dm26 = m_dm26; \
  Real_type dm27 = m_dm27; \
  Real_type dm28 = m_dm28; \
  Real_type c0 = m_c0; \
  const Index_type offset = m_offset; \
\
  allocAndInitCudaDeviceData(px, m_px, m_array_length);

#define INT_PREDICT_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_px, px, m_array_length); \
  deallocCudaDeviceData(px);

__global__ void int_predict(Real_ptr px,
                            Real_type dm22, Real_type dm23, Real_type dm24,
                            Real_type dm25, Real_type dm26, Real_type dm27,
                            Real_type dm28, Real_type c0,
                            const Index_type offset, 
                            Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     INT_PREDICT_BODY; 
   }
}


void INT_PREDICT::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_CUDA ) {

    INT_PREDICT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       int_predict<<<grid_size, block_size>>>( px, 
                                               dm22, dm23, dm24, dm25,
                                               dm26, dm27, dm28, c0,
                                               offset,
                                               iend ); 

    }
    stopTimer();

    INT_PREDICT_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    INT_PREDICT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         INT_PREDICT_BODY;
       });

    }
    stopTimer();

    INT_PREDICT_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  INT_PREDICT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
