  
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

#include "POLYBENCH_2MM.hpp"

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

#define POLYBENCH_2MM_DATA_SETUP_CUDA \
  Real_ptr tmp = m_tmp; \
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  Real_ptr C = m_C; \
  Real_ptr D = m_D; \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
  memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type)); \
  allocAndInitCudaDeviceData(tmp, m_tmp, m_ni * m_nj); \
  allocAndInitCudaDeviceData(A, m_A, m_ni * m_nk); \
  allocAndInitCudaDeviceData(B, m_B, m_nk * m_nj); \
  allocAndInitCudaDeviceData(C, m_C, m_nj * m_nl); \
  allocAndInitCudaDeviceData(D, m_D, m_ni * m_nl); 


#define POLYBENCH_2MM_TEARDOWN_CUDA \
  getCudaDeviceData(m_D, D, m_ni * m_nl); \
  deallocCudaDeviceData(tmp); \
  deallocCudaDeviceData(A); \
  deallocCudaDeviceData(B); \
  deallocCudaDeviceData(C); \
  deallocCudaDeviceData(D);

__global__ void polybench_2mm_cuda_1(Real_ptr tmp, Real_ptr A,
                       Real_ptr B, Real_ptr C, Real_ptr D,
                       Real_type alpha, Real_type beta, Index_type ni, Index_type nj,
                       Index_type nk, Index_type nl)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,j,k;
   if (ii < ni * nj) {
     *(tmp + ii) = 0.0;
     i = ii/nj; j = ii % nj;
     for (k=0; k < nk; k++) {
       POLYBENCH_2MM_BODY2;              
     }
   }


}

__global__ void polybench_2mm_cuda_2(Real_ptr tmp, Real_ptr A,
                       Real_ptr B, Real_ptr C, Real_ptr D,
                       Real_type alpha, Real_type beta, Index_type ni, Index_type nj,
                       Index_type nk, Index_type nl)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,l,j;
   if (ii < ni * nl) {
     *(D + ii) *= beta;
     i = ii/nl; l = ii % nl;
     for (j=0; j < nj; j++) {
       POLYBENCH_2MM_BODY4;              
     }
   }
}


void POLYBENCH_2MM::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ni = m_ni;
  const Index_type nj = m_nj;
  const Index_type nk = m_nk;
  const Index_type nl = m_nl;


  if ( vid == Base_CUDA ) {

    POLYBENCH_2MM_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nj, block_size);
      polybench_2mm_cuda_1<<<grid_size,block_size>>>(tmp,A,B,C,D,alpha,beta,
                                                     m_ni,m_nj,m_nk,m_nl);

      memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));
      initCudaDeviceData(D,m_D,m_ni * m_nl ); 

      grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nl, block_size);
      polybench_2mm_cuda_2<<<grid_size,block_size>>>(tmp,A,B,C,D,alpha,beta,
                                                     m_ni,m_nj,m_nk,m_nl);

    }
    stopTimer();

    POLYBENCH_2MM_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_2MM_DATA_SETUP_CUDA;

    using EXEC_POL = RAJA::nested::Policy<
                       RAJA::nested::CudaCollapse<
                         RAJA::nested::For<1, RAJA::cuda_block_y_exec>,   
                         RAJA::nested::For<0, RAJA::cuda_thread_x_exec> > >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
     
      RAJA::nested::forall(EXEC_POL{},
                           RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                            RAJA::RangeSegment(0, nj)),
        [=] __device__ (Index_type i, Index_type j) {

        POLYBENCH_2MM_BODY1;
        for (Index_type k=0;k<nk;k++) {
          POLYBENCH_2MM_BODY2; 
        }

      });

      memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));
      initCudaDeviceData(D,m_D,m_ni * m_nl ); 

      RAJA::nested::forall(EXEC_POL{},
                           RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                            RAJA::RangeSegment(0, nl)),
        [=] __device__ (Index_type i, Index_type l) {

        POLYBENCH_2MM_BODY3;
        for (Index_type j=0;j<nj;j++) {
          POLYBENCH_2MM_BODY4; 
        }

      });


    }
    stopTimer();

    POLYBENCH_2MM_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_2MM : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
  
