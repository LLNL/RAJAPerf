  
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

#include "POLYBENCH_GEMMVER.hpp"

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

#define POLYBENCH_GEMMVER_DATA_SETUP_CUDA \
  Index_type n = m_n; \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
  Real_ptr A = m_A; \
  Real_ptr u1 = m_u1; \
  Real_ptr v1 = m_v1; \
  Real_ptr u2 = m_u2; \
  Real_ptr v2 = m_v2; \
  Real_ptr w = m_w; \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_ptr z = m_z; \
\
  allocAndInitCudaDeviceData(A, m_A, m_n * m_n); \
  allocAndInitCudaDeviceData(u1, m_u1, m_n); \
  allocAndInitCudaDeviceData(v1, m_v1, m_n); \
  allocAndInitCudaDeviceData(u2, m_u2, m_n); \
  allocAndInitCudaDeviceData(v2, m_v2, m_n); \
  allocAndInitCudaDeviceData(w, m_w, m_n); \
  allocAndInitCudaDeviceData(x, m_x, m_n); \
  allocAndInitCudaDeviceData(y, m_y, m_n); \
  allocAndInitCudaDeviceData(z, m_z, m_n); 


#define POLYBENCH_GEMMVER_TEARDOWN_CUDA \
  getCudaDeviceData(m_w, w, m_n); \
  deallocCudaDeviceData(A); \
  deallocCudaDeviceData(u1); \
  deallocCudaDeviceData(v1); \
  deallocCudaDeviceData(u2); \
  deallocCudaDeviceData(v2); \
  deallocCudaDeviceData(w); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(z); 

__global__ void polybench_gemmver_cuda_1(Real_ptr A,
                       Real_ptr u1, Real_ptr v1, Real_ptr u2,
                       Real_ptr v2, Index_type n)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,j;
   if (ii < n * n) {
     i = ii/n; j = ii % n;
     POLYBENCH_GEMMVER_BODY1;
   }
}

__global__ void polybench_gemmver_cuda_2(Real_type beta,
                       Real_ptr A, Real_ptr x, Real_ptr y,
                       Index_type n)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,jj;
   if (ii < n * n) {
     i = ii/n; jj = ii % n;
     if(jj == 0) {
       for(Index_type j=0; j < n; ++j) { 
         POLYBENCH_GEMMVER_BODY2;
       } 
     }   
          
   }
}


__global__ void polybench_gemmver_cuda_3(Real_ptr x,
                       Real_ptr z, Real_ptr v1, Real_ptr u2,
                       Real_ptr v2, Index_type n)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) {
     POLYBENCH_GEMMVER_BODY3;              
   }
}

__global__ void polybench_gemmver_cuda_4(Real_type alpha,
                       Real_ptr A, Real_ptr x, Real_ptr w,
                       Index_type n)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,jj;
   if (ii < n * n) {
     i = ii/n; jj = ii % n;
     if(jj == 0) {
       for(Index_type j=0; j < n; ++j) { 
         POLYBENCH_GEMMVER_BODY4;
       } 
     }   
   }
}



void POLYBENCH_GEMMVER::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  
  if ( vid == Base_CUDA ) {

    POLYBENCH_GEMMVER_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_n * m_n, block_size);
      polybench_gemmver_cuda_1<<<grid_size,block_size>>>(A,u1,v1,u2,v2,n);

      grid_size = RAJA_DIVIDE_CEILING_INT(m_n * m_n, block_size);
      polybench_gemmver_cuda_2<<<grid_size,block_size>>>(beta,A,x,y,n);

      grid_size = RAJA_DIVIDE_CEILING_INT(m_n , block_size);
      polybench_gemmver_cuda_3<<<grid_size,block_size>>>(x,z,v1,u2,v2,n);

      grid_size = RAJA_DIVIDE_CEILING_INT(m_n * m_n, block_size);
      polybench_gemmver_cuda_4<<<grid_size,block_size>>>(alpha,A,x,w,n);

    }
    stopTimer();

    POLYBENCH_GEMMVER_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_GEMMVER_DATA_SETUP_CUDA;

    const bool async = true;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
     
      RAJA::forall<RAJA::cuda_exec<block_size, async>> (
        RAJA::RangeSegment{0, n * n}, [=] __device__ (int ii) {
        Index_type i,j;
        i = ii/n; j = ii % n;
        POLYBENCH_GEMMVER_BODY1; 
      });

      RAJA::forall<RAJA::cuda_exec<block_size, async>> (
        RAJA::RangeSegment{0, n * n}, [=] __device__ (int ii) {
          Index_type i,jj;
          i = ii/n; jj = ii % n;
          if(jj == 0) {
            for(Index_type j=0; j < n; ++j) { 
              POLYBENCH_GEMMVER_BODY2;
            } 
          }
      });

      RAJA::forall<RAJA::cuda_exec<block_size, async>> (
        RAJA::RangeSegment{0, n}, [=] __device__ (int i) {
        POLYBENCH_GEMMVER_BODY3;
      });

      RAJA::forall<RAJA::cuda_exec<block_size, async>> (
        RAJA::RangeSegment{0, n * n}, [=] __device__ (int ii) {
          Index_type i,jj;
          i = ii/n; jj = ii % n;
          if(jj == 0) {
            for(Index_type j=0; j < n; ++j) { 
              POLYBENCH_GEMMVER_BODY4;
            } 
          }   
      });
      
    }
    stopTimer();

    POLYBENCH_GEMMVER_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_GEMMVER : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
  
