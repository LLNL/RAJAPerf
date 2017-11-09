/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Polybench kernel ADI
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


#include "POLYBENCH_ADI.hpp"

#include "common/DataUtils.hpp"
#include <RAJA/RAJA.hpp>


#include <iostream>

namespace rajaperf 
{
namespace polybench
{



// The following ADI_BODY is a prototype of the kernel copied over from the polybench suite and is not used in the runKernel calls  
// It's just for illustrative purposes
#if 0
#pragma scop
  
  DX = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DY = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DT = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_TSTEPS;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 /  SCALAR_VAL(2.0);
  b = SCALAR_VAL(1.0)+mul1;
  c = a;
  d = -mul2 / SCALAR_VAL(2.0);
  e = SCALAR_VAL(1.0)+mul2;
  f = d;

 for (t=1; t<=_PB_TSTEPS; t++) {
    //Column Sweep
    for (i=1; i<_PB_N-1; i++) {
      v[0][i] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = v[0][i];
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -c / (a*p[i][j-1]+b);
        q[i][j] = (-d*u[j][i-1]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[j][i] - f*u[j][i+1]-a*q[i][j-1])/(a*p[i][j-1]+b);
      }
      
      v[_PB_N-1][i] = SCALAR_VAL(1.0);
      for (j=_PB_N-2; j>=1; j--) {
        v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
      }
    }
    //Row Sweep
    for (i=1; i<_PB_N-1; i++) {
      u[i][0] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = u[i][0];
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -f / (d*p[i][j-1]+e);
        q[i][j] = (-a*v[i-1][j]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][j] - c*v[i+1][j]-d*q[i][j-1])/(d*p[i][j-1]+e);
      }
      u[i][_PB_N-1] = SCALAR_VAL(1.0);
      for (j=_PB_N-2; j>=1; j--) {
        u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
      }
    }
  }

#pragma endscop
#endif


#define POLYBENCH_ADI_DATA \
  ResReal_ptr U = m_U; \
  ResReal_ptr V = m_V; \
  ResReal_ptr P = m_P; \
  ResReal_ptr Q = m_Q; \
  
#define POLYBENCH_ADI_BODY1 \
  Index_type t,i,j; \
  Real_type DX,DY,DT; \
  Real_type B1,B2; \
  Real_type mul1,mul2; \
  Real_type a,b,c,d,e,f; \
  DX = 1.0/(Real_type)m_n; \
  DY = 1.0/(Real_type)m_n; \
  DT = 1.0/(Real_type)m_tsteps; \
  B1 = 2.0; \
  B2 = 1.0; \
  mul1 = B1 * DT / (DX * DX); \
  mul2 = B2 * DT / (DY * DY); \
  a = -mul1 / 2.0; \
  b = 1.0 + mul1; \
  c = a; \
  d = -mul2 /2.0; \
  e = 1.0 + mul2; \
  f = d; 


#define POLYBENCH_ADI_BODY2 \
  *(V + 0 * n + i) = 1.0; \
  *(P + i * n + 0) = 0.0; \
  *(Q + i * n + 0) = *(V + 0 * n + i);

#define POLYBENCH_ADI_BODY3 \
  *(P + i * n + j) = -c / (a * *(P + i * n + j-1)+b); \
  *(Q + i * n + j) = (-d * *(U + j * n + i-1) + (1.0 + 2.0*d) * *(U + j * n + i) - f* *(U + j * n + i + 1) -a * *(Q + i * n + j-1))/(a * *(P + i * n + j -1)+b);

#define POLYBENCH_ADI_BODY4 \
  *(V + (n-1) * n + i) = 1.0;

#define POLYBENCH_ADI_BODY5 \
  *(V + j * n + i)  = *(P + i * n + j) * *(V + (j+1 * n) + i) + *(Q + i * n + j);

#define POLYBENCH_ADI_BODY6 \
  *(U + i * n + 0) = 1.0; \
  *(P + i * n + 0) = 0.0; \
  *(Q + i * n + 0) = *(U + i * n + 0);

#define POLYBENCH_ADI_BODY7 \
  *(P + i * n + j) = -f / (d * *(P + i * n + j-1)+e); \
  *(Q + i * n + j) = (-a * *(V + (i-1) * n + j) + (1.0 + 2.0*a) * *(V + i * n + j) - c * *(V + (i + 1) * n + j) -d * *(Q + i * n + j-1))/(d * *(P + i * n + j-1)+e);

#define POLYBENCH_ADI_BODY8 \
  *(U + i * n + n-1) = 1.0;

#define POLYBENCH_ADI_BODY9 \
  *(U + i * n + j)= *(P + i * n + j) * *(U + i * n + j +1) + *(Q + i * n + j);


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define POLYBENCH_ADI_DATA_SETUP_CUDA \
  ResReal_ptr U = m_U; \
  ResReal_ptr V = m_V; \
  ResReal_ptr P = m_P; \
  ResReal_ptr Q = m_Q; \
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


#if 0

__global__ void polybench_adi_cuda_1(Real_ptr A,
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
       POLYBENCH_ADI_BODY2;              
     }
   }
}

__global__ void polybench_adi_cuda_2(Real_ptr A,
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
       POLYBENCH_ADI_BODY4;              
     }
   }
}


__global__ void polybench_adi_cuda_3(Real_ptr A,
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
       POLYBENCH_ADI_BODY6;              
     }
   }
}

#endif
#endif // if defined(RAJA_ENABLE_CUDA)
  
POLYBENCH_ADI::POLYBENCH_ADI(const RunParams& params)
  : KernelBase(rajaperf::Polybench_ADI, params)
{
  SizeSpec_T lsizespec = KernelBase::getSizeSpec();
  switch(lsizespec) {
    case Mini:
      m_n=20; m_tsteps=20; 
      m_run_reps = 10000;
      break;
    case Small:
      m_n=60; m_tsteps=40; 
      m_run_reps = 500;
      break;
    case Medium:
      m_n=200; m_tsteps=100; 
      m_run_reps = 10;
      break;
    case Large:
      m_n=1000; m_tsteps=500; 
      m_run_reps = 1;
      break;
    case Extralarge:
      m_n=2000; m_tsteps=1000; 
      m_run_reps = 1;
      break;
    default:
      m_n=200; m_tsteps=100; 
      m_run_reps = 100;
      break;
  }
  setDefaultSize( m_tsteps );
  setDefaultReps(m_run_reps);
  allocAndInitData(m_U, m_n * m_n);
  allocAndInitData(m_V, m_n * m_n);
  allocAndInitData(m_P, m_n * m_n);
  allocAndInitData(m_Q, m_n * m_n);
}

POLYBENCH_ADI::~POLYBENCH_ADI() 
{
  deallocData(m_U);
  deallocData(m_V);
  deallocData(m_P);
  deallocData(m_Q);
}

void POLYBENCH_ADI::setUp(VariantID vid)
{

}

void POLYBENCH_ADI::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type n = m_n;
  const Index_type tsteps = m_tsteps;

  switch ( vid ) {

    case Base_Seq : {

      POLYBENCH_ADI_DATA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        POLYBENCH_ADI_BODY1;
        for (Index_type t = 1; t <= tsteps; t++ ) { 
          for(Index_type i = 1; i < n-1; i++) {
  //          POLYBENCH_ADI_BODY2;
            //*(V + (0 * n) + i) = 1.0; 
            *(P + i * n + 0) = 0.0; 
            *(Q + i * n + 0) = *(V + (0 * n) + i);
            for(Index_type j = 1; j < n-1; j++)
              POLYBENCH_ADI_BODY3;
            POLYBENCH_ADI_BODY4;
            for(j = n-2; j>=1; j--)
              POLYBENCH_ADI_BODY5;
          }
          for(Index_type i = 1; i < n-1; i++) {
            POLYBENCH_ADI_BODY6;
            for(Index_type j = 1; j < n-1; j++)
              POLYBENCH_ADI_BODY7;
            POLYBENCH_ADI_BODY8;
            for(j = n-2; j>=1; j--)
              POLYBENCH_ADI_BODY9;
          }
        }
      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {
      POLYBENCH_ADI_DATA;
      startTimer();
#if 0      
      for (SampIndex_type isamp = 0; isamp < m_run_samples; ++isamp) {
      
        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nj}, [=] (int i, int j) {
          POLYBENCH_ADI_BODY1;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nk}, [=] (int k) {
            POLYBENCH_ADI_BODY2; 
          });
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, nj}, RAJA::RangeSegment{0, nl}, [=] (int j, int l) {
          POLYBENCH_ADI_BODY3;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nm}, [=] (int m) {
            POLYBENCH_ADI_BODY4;
          });
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nl}, [=] (int i, int l) {
          POLYBENCH_ADI_BODY5;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nj}, [=] (int j) {
            POLYBENCH_ADI_BODY6;
          });
        });

      }
#endif     
      stopTimer();
      break;
    }

    case Base_OpenMP : {

#if defined(RAJA_ENABLE_OPENMP)      
      POLYBENCH_ADI_DATA;
      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < m_run_samples; ++isamp) {
        #pragma omp parallel for  
        for (Index_type i = 0; i < ni; i++ ) 
          for(Index_type j = 0; j < nj; j++) {
            POLYBENCH_ADI_BODY1;
            for(Index_type k = 0; k < nk; k++) {
              POLYBENCH_ADI_BODY2;
            }
          }

        #pragma omp parallel for   
        for(Index_type j = 0; j < nj; j++)
          for(Index_type l = 0; l < nl; l++) {
            POLYBENCH_ADI_BODY3;
            for(Index_type m = 0; m < nm; m++)
              POLYBENCH_ADI_BODY4;
          }

        #pragma omp parallel for   
        for(Index_type i = 0; i < ni; i++)
          for(Index_type l = 0; l < nl; l++) {
            POLYBENCH_ADI_BODY5;
            for(Index_type j = 0; j < nj; j++)
              POLYBENCH_ADI_BODY6;
          }  

      }
#endif
      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {
      // case is not defined...
      break;
    }

    case RAJA_OpenMP : {
      POLYBENCH_ADI_DATA;
      startTimer();
#if 0      
      for (SampIndex_type isamp = 0; isamp < m_run_samples; ++isamp) {
        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nj}, [=] (int i, int j) {
          POLYBENCH_ADI_BODY1;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nk}, [=] (int k) {
            POLYBENCH_ADI_BODY2; 
          });
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, nj}, RAJA::RangeSegment{0, nl}, [=] (int j, int l) {
          POLYBENCH_ADI_BODY3;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nm}, [=] (int m) {
            POLYBENCH_ADI_BODY4;
          });
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nl}, [=] (int i, int l) {
          POLYBENCH_ADI_BODY5;

          RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, nj}, [=] (int j) {
            POLYBENCH_ADI_BODY6;
          });
        });

      }
#endif
      stopTimer();
      break;
    }

#endif // RAJA_ENABLE_OPENMP

#if defined(RAJA_ENABLE_CUDA)
    case Baseline_CUDA : {
#if 0
      POLYBENCH_ADI_DATA_SETUP_CUDA;
      startTimer();
      for (SampIndex_type isamp = 0; isamp < m_run_samples; ++isamp) {
        size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nj, block_size);
        polybench_3mm_cuda_1<<<grid_size,block_size>>>(A,B,C,D,E,F,G,m_ni,m_nj,m_nk,m_nl,m_nm);

        grid_size = RAJA_DIVIDE_CEILING_INT(m_nj * m_nl, block_size);
        polybench_3mm_cuda_2<<<grid_size,block_size>>>(A,B,C,D,E,F,G,m_ni,m_nj,m_nk,m_nl,m_nm);

        grid_size = RAJA_DIVIDE_CEILING_INT(m_ni * m_nl, block_size);
        polybench_3mm_cuda_3<<<grid_size,block_size>>>(A,B,C,D,E,F,G,m_ni,m_nj,m_nk,m_nl,m_nm);
      }
      cudaDeviceSynchronize();
      stopTimer();
      POLYBENCH_ADI_TEARDOWN_CUDA;
#endif
      break;
    }

    case RAJA_CUDA : {
#if 0
      POLYBENCH_ADI_DATA_SETUP_CUDA;
      startTimer();
      for (SampIndex_type isamp = 0; isamp < m_run_samples; ++isamp) {
       
        RAJA::forall<RAJA::cuda_exec<block_size>> (RAJA::RangeSegment{0, ni * nj}, [=] __device__ (int ii) {
          Index_type i,j,k;
          *(E + ii) = 0.0;
          i = ii/nj; j = ii % nj;
          for(k=0;k<nk;k++) {
            POLYBENCH_ADI_BODY2; 
          }
        });

        RAJA::forall<RAJA::cuda_exec<block_size>> (RAJA::RangeSegment{0, nj * nl}, [=] __device__ (int ii) {
          *(F + ii) = 0.0;
          Index_type j,l,m;
          j = ii/nl; l = ii % nl;
          for(m=0;m<nm;m++) {
            POLYBENCH_ADI_BODY4;
          }  
        });

        RAJA::forall<RAJA::cuda_exec<block_size>> (RAJA::RangeSegment{0, ni * nl}, [=] __device__ (int ii) {
          *(G + ii) = 0.0;
          Index_type i,l,j;
          i = ii/nl; l = ii % nl;
          for(j=0;j<nj;j++) {
            POLYBENCH_ADI_BODY6;
          }  
        });

      }
      stopTimer();
      POLYBENCH_ADI_TEARDOWN_CUDA;
#endif
      break;
    }

#endif

#if 0
    case Baseline_OpenMP4x :
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

void POLYBENCH_ADI::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_U, m_n * m_n);
}

void POLYBENCH_ADI::tearDown(VariantID vid)
{
  (void) vid;

}

} // end namespace basic
} // end namespace rajaperf
