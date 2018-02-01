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

#include "INIT_VIEW1D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define INIT_VIEW1D_DATA_SETUP_CUDA \
  Real_ptr a; \
  const Real_type v = m_val; \
\
  allocAndInitCudaDeviceData(a, m_a, iend);

#define INIT_VIEW1D_DATA_RAJA_SETUP_CUDA \
  Real_ptr a; \
  const Real_type v = m_val; \
\
  allocAndInitCudaDeviceData(a, m_a, iend); \
\
  using ViewType = RAJA::View<Real_type, RAJA::Layout<1> >; \
  const RAJA::Layout<1> my_layout(iend); \
  ViewType view(a, my_layout);

#define INIT_VIEW1D_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_a, a, iend); \
  deallocCudaDeviceData(a);

__global__ void initview1d(Real_ptr a, 
                           Real_type v,
                           const Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     INIT_VIEW1D_BODY; 
   }
}


void INIT_VIEW1D::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_CUDA ) {

    INIT_VIEW1D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       initview1d<<<grid_size, block_size>>>( a,
                                              v, 
                                              iend ); 

    }
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    INIT_VIEW1D_DATA_RAJA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        INIT_VIEW1D_BODY_RAJA;
      });

    }
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  INIT_VIEW1D : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
