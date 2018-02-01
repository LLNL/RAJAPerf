  
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

#include "POLYBENCH_3MM.hpp"

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

#define POLYBENCH_3MM_DATA_SETUP_CUDA \
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  Real_ptr C = m_C; \
  Real_ptr D = m_D; \
  Real_ptr E = m_E; \
  Real_ptr F = m_F; \
  Real_ptr G = m_G; \
\
  allocAndInitCudaDeviceData(A, m_A, m_ni * m_nk); \
  allocAndInitCudaDeviceData(B, m_B, m_nk * m_nj); \
  allocAndInitCudaDeviceData(C, m_C, m_nj * m_nm); \
  allocAndInitCudaDeviceData(D, m_D, m_nm * m_nl); \
  allocAndInitCudaDeviceData(E, m_E, m_ni * m_nj); \
  allocAndInitCudaDeviceData(F, m_F, m_nj * m_nl); \
  allocAndInitCudaDeviceData(G, m_G, m_ni * m_nl); 


#define POLYBENCH_3MM_TEARDOWN_CUDA \
  getCudaDeviceData(m_G, G, m_ni * m_nl); \
  deallocCudaDeviceData(A); \
  deallocCudaDeviceData(B); \
  deallocCudaDeviceData(C); \
  deallocCudaDeviceData(D); \
  deallocCudaDeviceData(E); \
  deallocCudaDeviceData(F); \
  deallocCudaDeviceData(G);

__global__ void polybench_3mm_cuda_1(Real_ptr A,
                       Real_ptr B, Real_ptr C, Real_ptr D,
                       Real_ptr E, Real_ptr F, Real_ptr G,
                       Index_type ni, Index_type nj,
                       Index_type nk, Index_type nl, Index_type nm)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,j,k;
   if (ii < ni * nj) {
     *(E + ii) = 0.0;
     i = ii/nj; j = ii % nj;
     for (k=0; k < nk; k++) {
       POLYBENCH_3MM_BODY2;              
     }
   }
}

__global__ void polybench_3mm_cuda_2(Real_ptr A,
                       Real_ptr B, Real_ptr C, Real_ptr D,
                       Real_ptr E, Real_ptr F, Real_ptr G,
                       Index_type ni, Index_type nj,
                       Index_type nk, Index_type nl, Index_type nm)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type j,l,m;
   if (ii < nj * nl) {
     *(F + ii) = 0.0;
     j = ii/nl; l = ii % nl;
     for (m=0; m < nm; m++) {
       POLYBENCH_3MM_BODY4;              
     }
   }
}


__global__ void polybench_3mm_cuda_3(Real_ptr A,
                       Real_ptr B, Real_ptr C, Real_ptr D,
                       Real_ptr E, Real_ptr F, Real_ptr G,
                       Index_type ni, Index_type nj,
                       Index_type nk, Index_type nl, Index_type nm)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i,l,j;
   if (ii < ni * nl) {
     *(G + ii) = 0.0;
     i = ii/nl; l = ii % nl;
     for (j=0; j < nj; j++) {
       POLYBENCH_3MM_BODY6;              
     }
   }
}

void POLYBENCH_3MM::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ni = m_ni;
  const Index_type nj = m_nj;
  const Index_type nk = m_nk;
  const Index_type nl = m_nl;
  const Index_type nm = m_nm;

  
  if ( vid == Base_CUDA ) {

    POLYBENCH_3MM_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nj, block_size);
      polybench_3mm_cuda_1<<<grid_size,block_size>>>(A,B,C,D,E,F,G,
                                                     m_ni,m_nj,m_nk,m_nl,m_nm);

      grid_size = RAJA_DIVIDE_CEILING_INT(m_nj * m_nl, block_size);
      polybench_3mm_cuda_2<<<grid_size,block_size>>>(A,B,C,D,E,F,G,
                                                     m_ni,m_nj,m_nk,m_nl,m_nm);

      grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nl, block_size);
      polybench_3mm_cuda_3<<<grid_size,block_size>>>(A,B,C,D,E,F,G,
                                                     m_ni,m_nj,m_nk,m_nl,m_nm);
    }
    stopTimer();

    
    POLYBENCH_3MM_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_3MM_DATA_SETUP_CUDA;

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

        POLYBENCH_3MM_BODY1;
        for (Index_type k=0;k<nk;k++) {
          POLYBENCH_3MM_BODY2; 
        }

      });

      RAJA::nested::forall(EXEC_POL{},
                           RAJA::make_tuple(RAJA::RangeSegment(0, nj),
                                            RAJA::RangeSegment(0, nl)),
        [=] __device__ (Index_type j, Index_type l) {

        POLYBENCH_3MM_BODY3;
        for (Index_type m=0;m<nm;m++) {
          POLYBENCH_3MM_BODY4; 
        }

      });

      RAJA::nested::forall(EXEC_POL{},
                           RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                            RAJA::RangeSegment(0, nl)),
        [=] __device__ (Index_type i, Index_type l) {

        POLYBENCH_3MM_BODY5;
        for (Index_type j=0;j<nj;j++) {
          POLYBENCH_3MM_BODY6; 
        }

      });

    }
    stopTimer();

    POLYBENCH_3MM_TEARDOWN_CUDA;

  } else {
      std::cout << "\n  POLYBENCH_3MM : Unknown Cuda variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
  
