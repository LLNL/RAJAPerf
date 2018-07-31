  
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

#include "POLYBENCH_ADI.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

//
// Define thread block size for CUDA execution
//
const size_t block_size = 256;

#define POLYBENCH_ADI_DATA_SETUP_CUDA \
  Real_ptr U = m_U; \
  Real_ptr V = m_V; \
  Real_ptr P = m_P; \
  Real_ptr Q = m_Q; \
\
  allocAndInitCudaDeviceData(U, m_U, m_n * m_n); \
  allocAndInitCudaDeviceData(V, m_V, m_n * m_n); \
  allocAndInitCudaDeviceData(P, m_P, m_n * m_n); \
  allocAndInitCudaDeviceData(Q, m_Q, m_n * m_n); 


#define POLYBENCH_ADI_TEARDOWN_CUDA \
  getCudaDeviceData(m_U, U, m_n * m_n); \
  deallocCudaDeviceData(U); \
  deallocCudaDeviceData(V); \
  deallocCudaDeviceData(P); \
  deallocCudaDeviceData(Q); 


__global__ void polybench_adi_cuda(Real_ptr U,
                       Real_ptr V, Real_ptr P, Real_ptr Q,
                       Real_type a, Real_type b, Real_type c,
                       Real_type d, Real_type e, Real_type f,
                       Index_type n)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type j;
   if (i < n - 1 && i > 0) {
     POLYBENCH_ADI_BODY2;
     for(j = 1; j < n-1; j++) {
       POLYBENCH_ADI_BODY3;
     }  
     POLYBENCH_ADI_BODY4;
     for(j = 1; j < n-1; j++) {
       POLYBENCH_ADI_BODY5;
     }  
   }
   __syncthreads();
   if (i < n - 1 && i > 0) {
     POLYBENCH_ADI_BODY6;
     for(j = 1; j < n-1; j++) {
       POLYBENCH_ADI_BODY7;
     }  
     POLYBENCH_ADI_BODY8;
     for(j = 1; j < n-1; j++) {
       POLYBENCH_ADI_BODY9;
     }  
   }
}


void POLYBENCH_ADI::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type n = m_n;
  const Index_type tsteps = m_tsteps;
  
  if ( vid == Base_CUDA ) {

    POLYBENCH_ADI_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      POLYBENCH_ADI_BODY1;
      for (Index_type t = 1; t <= tsteps; t++ ) { 
        size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_n * m_n, block_size);
        polybench_adi_cuda<<<grid_size,block_size>>>(U,V,P,Q,a,b,c,d,e,f,n);
      }  
    }
    stopTimer();

    POLYBENCH_ADI_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {
    POLYBENCH_ADI_DATA_SETUP_CUDA;

    const bool async = false;
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      POLYBENCH_ADI_BODY1;
//     RAJA::forall<RAJA::seq_exec> (
//        RAJA::RangeSegment(1, tsteps+1), [=](Index_type t) { 
        for (Index_type t = 1; t <= tsteps; t++ ) { 

          RAJA::forall<RAJA::cuda_exec<block_size, async>> (
            RAJA::RangeSegment{1, n - 1}, [=] __device__ (int i) {
              Index_type j;
              POLYBENCH_ADI_BODY2;
              for(j = 1; j < n-1; j++) {
                POLYBENCH_ADI_BODY3;
              }  
              POLYBENCH_ADI_BODY4;
              for(j = 1; j < n-1; j++) {
                POLYBENCH_ADI_BODY5;
              }  
            });
          RAJA::forall<RAJA::cuda_exec<block_size, async>> (
            RAJA::RangeSegment{1, n - 1}, [=] __device__ (int i) {
              Index_type j;
              POLYBENCH_ADI_BODY6;
              for(j = 1; j < n-1; j++) {
                POLYBENCH_ADI_BODY7;
              }  
              POLYBENCH_ADI_BODY8;
              for(j = 1; j < n-1; j++) {
                POLYBENCH_ADI_BODY9;
              }  
            });
        } //Tsteps  
//        }); // tsteps
    }

    stopTimer();

    POLYBENCH_ADI_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_ADI : Unknown Cuda variant id = " << vid << std::endl;
  }
}
  
} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
  
