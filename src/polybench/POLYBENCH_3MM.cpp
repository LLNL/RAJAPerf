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
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "POLYBENCH_3MM.hpp"

#include "common/DataUtils.hpp"
#include <RAJA/RAJA.hpp>


#include <iostream>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_3MM_DATA \
  ResReal_ptr A = m_A; \
  ResReal_ptr B = m_B; \
  ResReal_ptr C = m_C; \
  ResReal_ptr D = m_D; \
  ResReal_ptr E = m_E; \
  ResReal_ptr F = m_F; \
  ResReal_ptr G = m_G; 
  


// The following 3MM_BODY is a prototype of the kernel copied over from the polybench suite and is not used in the runKernel calls  
// It's just for illustrative purposes
#if 0
#pragma scop
  /* E := A*B */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NJ; j++)
      {
	      E[i][j] = SCALAR_VAL(0.0);
	      for (k = 0; k < _PB_NK; ++k)
	        E[i][j] += A[i][k] * B[k][j];
      }
  /* F := C*D */
  for (i = 0; i < _PB_NJ; i++)
    for (j = 0; j < _PB_NL; j++)
      {
	      F[i][j] = SCALAR_VAL(0.0);
	      for (k = 0; k < _PB_NM; ++k)
	        F[i][j] += C[i][k] * D[k][j];
      }
  /* G := E*F */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NL; j++)
      {
	      G[i][j] = SCALAR_VAL(0.0);
	      for (k = 0; k < _PB_NJ; ++k)
	        G[i][j] += E[i][k] * F[k][j];
      }
#pragma endscop
#endif

#define POLYBENCH_3MM_BODY1 \
  *(E + i * nj + j) = 0.0;

#define POLYBENCH_3MM_BODY2 \
  *(E + i * nj + j) += *(A + i * nk + k) * *(B + k * nj + j);

#define POLYBENCH_3MM_BODY3 \
  *(F + j * nl + l) = 0.0;

#define POLYBENCH_3MM_BODY4 \
  *(F + j * nl + l)  += *(C + j * nm + m) * *(D + m * nl + l);

#define POLYBENCH_3MM_BODY5 \
  *(G + i * nl + l) = 0.0;

#define POLYBENCH_3MM_BODY6 \
  *(G + i * nl + l) += *(E + i * nj + j) * *(F + j * nl + l);


#if defined(ENABLE_CUDA)

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
     for(k=0; k < nk; k++) {
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
     for(m=0; m < nm; m++) {
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
     for(j=0; j < nj; j++) {
       POLYBENCH_3MM_BODY6;              
     }
   }
}


#endif // if defined(RAJA_ENABLE_CUDA)
  
POLYBENCH_3MM::POLYBENCH_3MM(const RunParams& params)
  : KernelBase(rajaperf::Polybench_3MM, params)
{
  setDefaultReps(1);
  SizeSpec_T lsizespec = KernelBase::getSizeSpec();
  switch(lsizespec) {
    case Mini:
      m_ni=16; m_nj=18; m_nk=20; m_nl=22; m_nm=24;
      m_run_reps = 100000;
      break;
    case Small:
      m_ni=40; m_nj=50; m_nk=60; m_nl=70; m_nm=80;
      m_run_reps = 5000;
      break;
    case Medium:
      m_ni=180; m_nj=190; m_nk=200; m_nl=210; m_nm=220;
      m_run_reps = 100;
      break;
    case Large:
      m_ni=800; m_nj=900; m_nk=1000; m_nl=1100; m_nm=1200;
      m_run_reps = 1;
      break;
    case Extralarge:
      m_ni=1600; m_nj=1800; m_nk=2000; m_nl=2200; m_nm=2400;
      m_run_reps = 1;
      break;
    default:
      m_ni=180; m_nj=190; m_nk=200; m_nl=210; m_nm=220;
      m_run_reps = 100;
      break;
  }
  setDefaultReps(m_run_reps);
  allocAndInitData(m_A, m_ni * m_nk);
  allocAndInitData(m_B, m_nk * m_nj);
  allocAndInitData(m_C, m_nj * m_nm);
  allocAndInitData(m_D, m_nm * m_nl);
  allocAndInitData(m_E, m_ni * m_nj);
  allocAndInitData(m_F, m_nj * m_nl);
  allocAndInitData(m_G, m_ni * m_nl);
}

POLYBENCH_3MM::~POLYBENCH_3MM() 
{
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_C);
  deallocData(m_D);
  deallocData(m_E);
  deallocData(m_F);
  deallocData(m_G);
}

void POLYBENCH_3MM::setUp(VariantID vid)
{
  (void) vid;
}

void POLYBENCH_3MM::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ni = m_ni;
  const Index_type nj = m_nj;
  const Index_type nk = m_nk;
  const Index_type nl = m_nl;
  const Index_type nm = m_nm;

  switch ( vid ) {

    case Base_Seq : {

      POLYBENCH_3MM_DATA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        for (Index_type i = 0; i < ni; i++ ) 
          for(Index_type j = 0; j < nj; j++) {
            POLYBENCH_3MM_BODY1;
            for(Index_type k = 0; k < nk; k++)
              POLYBENCH_3MM_BODY2;
          }

        for(Index_type j = 0; j < nj; j++)
          for(Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY3;
            for(Index_type m = 0; m < nm; m++)
              POLYBENCH_3MM_BODY4;
          }

        for(Index_type i = 0; i < ni; i++)
          for(Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY5;
            for(Index_type j = 0; j < nj; j++)
              POLYBENCH_3MM_BODY6;
          }
      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {
      POLYBENCH_3MM_DATA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nj}, [=] (int i, int j) {
          POLYBENCH_3MM_BODY1;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nk}, [=] (int k) {
            POLYBENCH_3MM_BODY2; 
          });
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, nj}, RAJA::RangeSegment{0, nl}, [=] (int j, int l) {
          POLYBENCH_3MM_BODY3;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nm}, [=] (int m) {
            POLYBENCH_3MM_BODY4;
          });
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nl}, [=] (int i, int l) {
          POLYBENCH_3MM_BODY5;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nj}, [=] (int j) {
            POLYBENCH_3MM_BODY6;
          });
        });

      }
      stopTimer();
      break;
    }

    case Base_OpenMP : {

#if defined(ENABLE_OPENMP)      
      POLYBENCH_3MM_DATA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        #pragma omp parallel for  
        for (Index_type i = 0; i < ni; i++ ) 
          for(Index_type j = 0; j < nj; j++) {
            POLYBENCH_3MM_BODY1;
            for(Index_type k = 0; k < nk; k++) {
              POLYBENCH_3MM_BODY2;
            }
          }

        #pragma omp parallel for   
        for(Index_type j = 0; j < nj; j++)
          for(Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY3;
            for(Index_type m = 0; m < nm; m++)
              POLYBENCH_3MM_BODY4;
          }

        #pragma omp parallel for   
        for(Index_type i = 0; i < ni; i++)
          for(Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY5;
            for(Index_type j = 0; j < nj; j++)
              POLYBENCH_3MM_BODY6;
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
#if defined(ENABLE_OPENMP)      
      POLYBENCH_3MM_DATA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nj}, [=] (int i, int j) {
          POLYBENCH_3MM_BODY1;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nk}, [=] (int k) {
            POLYBENCH_3MM_BODY2; 
          });
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, nj}, RAJA::RangeSegment{0, nl}, [=] (int j, int l) {
          POLYBENCH_3MM_BODY3;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nm}, [=] (int m) {
            POLYBENCH_3MM_BODY4;
          });
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nl}, [=] (int i, int l) {
          POLYBENCH_3MM_BODY5;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nj}, [=] (int j) {
            POLYBENCH_3MM_BODY6;
          });
        });

      }
      stopTimer();
#endif
      break;
    }

#if defined(ENABLE_CUDA)
    case Base_CUDA : {
      POLYBENCH_3MM_DATA_SETUP_CUDA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nj, block_size);
        polybench_3mm_cuda_1<<<grid_size,block_size>>>(A,B,C,D,E,F,G,m_ni,m_nj,m_nk,m_nl,m_nm);

        grid_size = RAJA_DIVIDE_CEILING_INT(m_nj * m_nl, block_size);
        polybench_3mm_cuda_2<<<grid_size,block_size>>>(A,B,C,D,E,F,G,m_ni,m_nj,m_nk,m_nl,m_nm);

        grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nl, block_size);
        polybench_3mm_cuda_3<<<grid_size,block_size>>>(A,B,C,D,E,F,G,m_ni,m_nj,m_nk,m_nl,m_nm);
      }
      cudaDeviceSynchronize();
      stopTimer();
      POLYBENCH_3MM_TEARDOWN_CUDA;
      break;
    }

    case RAJA_CUDA : {
      POLYBENCH_3MM_DATA_SETUP_CUDA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
       
        RAJA::forall<RAJA::cuda_exec<block_size>> (RAJA::RangeSegment{0, ni * nj}, [=] __device__ (int ii) {
          Index_type i,j,k;
          *(E + ii) = 0.0;
          i = ii/nj; j = ii % nj;
          for(k=0;k<nk;k++) {
            POLYBENCH_3MM_BODY2; 
          }
        });

        RAJA::forall<RAJA::cuda_exec<block_size>> (RAJA::RangeSegment{0, nj * nl}, [=] __device__ (int ii) {
          *(F + ii) = 0.0;
          Index_type j,l,m;
          j = ii/nl; l = ii % nl;
          for(m=0;m<nm;m++) {
            POLYBENCH_3MM_BODY4;
          }  
        });

        RAJA::forall<RAJA::cuda_exec<block_size>> (RAJA::RangeSegment{0, ni * nl}, [=] __device__ (int ii) {
          *(G + ii) = 0.0;
          Index_type i,l,j;
          i = ii/nl; l = ii % nl;
          for(j=0;j<nj;j++) {
            POLYBENCH_3MM_BODY6;
          }  
        });

      }
      stopTimer();
      POLYBENCH_3MM_TEARDOWN_CUDA;
      break;
    }

#endif

#if 0
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget : {
      // Fill these in later...you get the idea...
      break;
    }
#endif

    default : {
      std::cout << "\n  Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_3MM::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_G, m_ni * m_nl);
}

void POLYBENCH_3MM::tearDown(VariantID vid)
{
  (void) vid;

}

} // end namespace basic
} // end namespace rajaperf
