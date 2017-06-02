/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Polybench kernel 2MM
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-xxxxxx
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For additional details, please read the file LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "POLYBENCH_2MM.hpp"

#include "common/DataUtils.hpp"
#include <RAJA/RAJA.hpp>


#include <iostream>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_2MM_DATA \
  ResReal_ptr tmp = m_tmp; \
  ResReal_ptr A = m_A; \
  ResReal_ptr B = m_B; \
  ResReal_ptr C = m_C; \
  ResReal_ptr D = m_D; \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
  alpha = 1.5; \
  beta = 1.2; 


// The following 2MM_BODY is a prototype of the kernel copied over from the polybench suite and is not used in the runKernel calls  
// It's just for illustrative purposes
#define POLYBENCH_2MM_BODY \
  int i, j, k;\
  /* D := alpha*A*B*C + beta*D */ \
  for (i = 0; i < m_ni; i++) \
    for (j = 0; j < m_nj; j++) { \
	      m_tmp[i][j] = 0.0; \
	      for (k = 0; k < m_nk; ++k) \
	        m_tmp[i][j] += m_alpha * m_A[i][k] * m_B[k][j]; \
    } \
  for (i = 0; i < m_ni; i++) \
    for (j = 0; j < m_nl; j++) { \
	    m_D[i][j] *= m_beta; \
	    for (k = 0; k < m_nj; ++k) \
	      m_D[i][j] += m_tmp[i][k] * m_C[k][j]; \
    } 

#define POLYBENCH_2MM_BODY1 \
  *(tmp + i * nj + j) = 0.0;

#define POLYBENCH_2MM_BODY2 \
  *(tmp + i * nj + j) += alpha * *(A + i * nk + k) * *(B + k * nj + j);

#define POLYBENCH_2MM_BODY3 \
  *(D + i * nl + l) *= beta;

#define POLYBENCH_2MM_BODY4 \
  *(D + i * nl + l) += *(tmp + i * nj + j) * *(C + j * nl + l);


#if defined(RAJA_ENABLE_CUDA)

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
  alpha = 1.5; \
  beta = 1.2; \
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
     for(k=0; k < nk; k++) {
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
     for(j=0; j < nj; j++) {
       POLYBENCH_2MM_BODY4;              
     }
   }
}


#endif // if defined(RAJA_ENABLE_CUDA)
  
POLYBENCH_2MM::POLYBENCH_2MM(const RunParams& params)
  : KernelBase(rajaperf::Polybench_2MM, params)
{
  setDefaultReps(1);
  m_alpha = 1.5;
  m_beta = 1.2;
  SizeSpec_T lsizespec = KernelBase::getSizeSpec();
  switch(lsizespec) {
    case Mini:
      m_ni=16; m_nj=18; m_nk=22; m_nl=24;
      m_run_reps = 100000;
      break;
    case Small:
      m_ni=40; m_nj=50; m_nk=70; m_nl=80;
      m_run_reps = 10000;
      break;
    case Medium:
      m_ni=180; m_nj=190; m_nk=210; m_nl=220;
      m_run_reps = 100;
      break;
    case Large:
      m_ni=800; m_nj=900; m_nk=1100; m_nl=1200;
      m_run_reps = 1;
      break;
    case Extralarge:
      m_ni=1600; m_nj=1800; m_nk=2200; m_nl=2400;
      m_run_reps = 1;
      break;
    default:
      m_ni=180; m_nj=190; m_nk=210; m_nl=220;
      m_run_reps = 100;
      break;
  }

  setDefaultReps(m_run_reps);
  fprintf(stderr,"Polybench_2MM will run %d reps\n",getRunReps());
  allocAndInitData(m_tmp, m_ni * m_nj);
  allocAndInitData(m_A, m_ni * m_nk);
  allocAndInitData(m_B, m_nk * m_nj);
  allocAndInitData(m_C, m_nj * m_nl);
  allocAndInitData(m_D, m_ni * m_nl);
  allocAndInitData(m_DD, m_ni * m_nl);

}

POLYBENCH_2MM::~POLYBENCH_2MM() 
{
  deallocData(m_tmp);
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_C);
  deallocData(m_D);
  deallocData(m_DD);
}

void POLYBENCH_2MM::setUp(VariantID vid)
{

}

void POLYBENCH_2MM::runKernel(VariantID vid)
{
  const Index_type run_reps= getRunReps();
  const Index_type ni = m_ni;
  const Index_type nj = m_nj;
  const Index_type nk = m_nk;
  const Index_type nl = m_nl;

  switch ( vid ) {

    case Base_Seq : {

      POLYBENCH_2MM_DATA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        for (Index_type i = 0; i < ni; i++ ) 
          for(Index_type j = 0; j < nj; j++) {
            POLYBENCH_2MM_BODY1;
            for(Index_type k = 0; k < nk; k++)
              POLYBENCH_2MM_BODY2;
          }
        memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));
        for(Index_type i = 0; i < ni; i++)
          for(Index_type l = 0; l < nl; l++) {
            POLYBENCH_2MM_BODY3;
            for(Index_type j = 0; j < nj; j++)
              POLYBENCH_2MM_BODY4;
          }
      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      POLYBENCH_2MM_DATA;
      resetTimer();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {      

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nj}, [=] (int i, int j) {
          POLYBENCH_2MM_BODY1;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nk}, [=] (int k) {
            POLYBENCH_2MM_BODY2; 
          });
        });

        memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nl}, [=] (int i, int l) {
          POLYBENCH_2MM_BODY3;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nj}, [=] (int j) {
            POLYBENCH_2MM_BODY4;
          });
        });

      }
      stopTimer();

      break;
    }

    case Base_OpenMP : {
#if defined(_OPENMP)      
      POLYBENCH_2MM_DATA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        #pragma omp parallel for  
        for (Index_type i = 0; i < ni; i++ ) 
          for(Index_type j = 0; j < nj; j++) {
            //if(j == 0) printf("Thread : %d\n",omp_get_thread_num());
            POLYBENCH_2MM_BODY1;
            for(Index_type k = 0; k < nk; k++) {

              POLYBENCH_2MM_BODY2;
            }
          }


        memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));
        #pragma omp parallel for   
        for(Index_type i = 0; i < ni; i++)
          for(Index_type l = 0; l < nl; l++) {
            //if(l == 0) printf("Thread : %d\n",omp_get_thread_num());
            POLYBENCH_2MM_BODY3;
            for(Index_type j = 0; j < nj; j++)
              POLYBENCH_2MM_BODY4;
          }  

      }
      stopTimer();
#endif
      break;
    }

    case RAJALike_OpenMP : {
      // case is not defined...
      break;
    }

    case RAJA_OpenMP : {
#if defined(_OPENMP)      
      POLYBENCH_2MM_DATA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nj}, [=] (int i, int j) {
          POLYBENCH_2MM_BODY1;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nk}, [=] (int k) {
            POLYBENCH_2MM_BODY2; 
          });
        });
        memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));
        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nl}, [=] (int i, int l) {
          POLYBENCH_2MM_BODY3;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nj}, [=] (int j) {
            POLYBENCH_2MM_BODY4;
          });
        });


      }
      stopTimer();
#endif
      break;
    }

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {

      POLYBENCH_2MM_DATA_SETUP_CUDA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nj, block_size);
        polybench_2mm_cuda_1<<<grid_size,block_size>>>(tmp,A,B,C,D,alpha,beta,m_ni,m_nj,m_nk,m_nl);

        memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));
        initCudaDeviceData(D,m_D,m_ni * m_nl ); 

        grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nl, block_size);
        polybench_2mm_cuda_2<<<grid_size,block_size>>>(tmp,A,B,C,D,alpha,beta,m_ni,m_nj,m_nk,m_nl);
      }
      cudaDeviceSynchronize();
      stopTimer();
      POLYBENCH_2MM_TEARDOWN_CUDA;
      break;
    }

    case RAJA_CUDA : {

      POLYBENCH_2MM_DATA_SETUP_CUDA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
       
        RAJA::forall<RAJA::cuda_exec<block_size>> (RAJA::RangeSegment{0, ni * nj}, [=] __device__ (int ii) {
          Index_type i,j,k;
          *(tmp + ii) = 0.0;
          i = ii/nj; j = ii % nj;
          for(k=0;k<nk;k++) {
            POLYBENCH_2MM_BODY2; 
          }
        });

        memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));
        initCudaDeviceData(D,m_D,m_ni * m_nl ); 

        RAJA::forall<RAJA::cuda_exec<block_size>> (RAJA::RangeSegment{0, ni * nl}, [=] __device__ (int ii) {
          *(D + ii) *= beta;
          Index_type i,l,j;
          i = ii/nl; l = ii % nl;
          for(j=0;j<nj;j++) {
            POLYBENCH_2MM_BODY4;
          }  
        });

      }
      stopTimer();
      POLYBENCH_2MM_TEARDOWN_CUDA;
      break;
    }

#endif

#if 0
    case Base_OpenMP4x :
    case RAJA_OpenMP4x : {
      // Fill these in later...you get the idea...
      break;
    }
#endif

    default : {
      std::cout << "\n  Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_2MM::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_D, m_ni * m_nl);
}

void POLYBENCH_2MM::tearDown(VariantID vid)
{
  (void) vid;

}

} // end namespace basic
} // end namespace rajaperf
