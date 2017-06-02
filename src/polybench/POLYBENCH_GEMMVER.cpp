/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Polybench kernel GEMMVER
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


#include "POLYBENCH_GEMMVER.hpp"

#include "common/DataUtils.hpp"
#include <RAJA/RAJA.hpp>


#include <iostream>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_GEMMVER_DATA \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
  ResReal_ptr A = m_A; \
  ResReal_ptr u1 = m_u1; \
  ResReal_ptr v1 = m_v1; \
  ResReal_ptr u2 = m_u2; \
  ResReal_ptr v2 = m_v2; \
  ResReal_ptr w = m_w; \
  ResReal_ptr x = m_x; \
  ResReal_ptr y = m_y; \
  ResReal_ptr z = m_z; 
  


// The following GEMMVER_BODY is a prototype of the kernel copied over from the polybench suite and is not used in the runKernel calls  
// It's just for illustrative purposes
#if 0
#pragma scop
  
 
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < _PB_N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];


#pragma endscop
#endif

#define POLYBENCH_GEMMVER_BODY1 \
  *(A + i * n + j) = *(A + i * n + j) + *(u1 + i) * *(v1 + j) + *(u2 + i) * *(v2 + j)

#define POLYBENCH_GEMMVER_BODY2 \
  *(x + i) = *(x + i) + beta * *(A + j * n + i) * *(y + j);

#define POLYBENCH_GEMMVER_BODY3 \
  *(x + i) = *(x + i) + *(z + i);

#define POLYBENCH_GEMMVER_BODY4 \
  *(w + i) = *(w + i) + alpha * *(A + i * n + j) * *(x + j);



#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define POLYBENCH_GEMMVER_DATA_SETUP_CUDA \
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
   Index_type i,j;
   if (ii < n * n) {
     i = ii/n; j = ii % n;
     POLYBENCH_GEMMVER_BODY2;              
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
   Index_type i,j;
   if (ii < n * n) {
     i = ii/n; j = ii % n;
     POLYBENCH_GEMMVER_BODY4;              
   }
}



#endif // if defined(RAJA_ENABLE_CUDA)
  
POLYBENCH_GEMMVER::POLYBENCH_GEMMVER(const RunParams& params)
  : KernelBase(rajaperf::Polybench_GEMMVER, params)
{
  //setDefaultReps(2000);

  SizeSpec_T lsizespec = KernelBase::getSizeSpec();
  switch(lsizespec) {
    case Mini:
      m_n=40;
      m_run_reps = 200000;
      break;
    case Small:
      m_n=120; 
      m_run_reps = 20000;
      break;
    case Medium:
      m_n=400;
      m_run_reps = 2000;
      break;
    case Large:
      m_n=2000;
      m_run_reps = 20;
      break;
    case Extralarge:
      m_n=4000; 
      m_run_reps = 5;
      break;
    default:
      m_n=400;
      m_run_reps = 2000;
      break;
  }

  setDefaultReps(m_run_reps);
  fprintf(stderr,"Polybench_GEMMVAR will run %d reps\n",getRunReps());
  allocAndInitData(m_A, m_n * m_n);
  allocAndInitData(m_u1, m_n);
  allocAndInitData(m_v1, m_n);
  allocAndInitData(m_u2, m_n);
  allocAndInitData(m_v2, m_n);
  allocAndInitData(m_w, m_n);
  allocAndInitData(m_x, m_n);
  allocAndInitData(m_y, m_n);
  allocAndInitData(m_z, m_n);
}

POLYBENCH_GEMMVER::~POLYBENCH_GEMMVER() 
{
  deallocData(m_A);
  deallocData(m_u1);
  deallocData(m_v1);
  deallocData(m_u2);
  deallocData(m_v2);
  deallocData(m_w);
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_z);
}

void POLYBENCH_GEMMVER::setUp(VariantID vid)
{

}

void POLYBENCH_GEMMVER::runKernel(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type n = m_n;

  switch ( vid ) {

    case Base_Seq : {

      POLYBENCH_GEMMVER_DATA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        for (Index_type i = 0; i < n; i++ ) 
          for(Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY1;
          }

        for (Index_type i = 0; i < n; i++ ) 
          for(Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY2;
          }

        for (Index_type i = 0; i < n; i++ ) 
          for(Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY3;
          }

        for (Index_type i = 0; i < n; i++ ) 
          for(Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY4;
          }
      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {
      POLYBENCH_GEMMVER_DATA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=] (int i, int j) {
          POLYBENCH_GEMMVER_BODY1;
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=] (int i, int j) {
          POLYBENCH_GEMMVER_BODY2;
        });


        RAJA::forall<RAJA::seq_exec> (RAJA::RangeSegment{0, n}, [=] (int i) {
            POLYBENCH_GEMMVER_BODY3; 
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=] (int i, int j) {
          POLYBENCH_GEMMVER_BODY4;
        });

      }
      stopTimer();
      break;
    }

    case Base_OpenMP : {

#if defined(_OPENMP)      
      POLYBENCH_GEMMVER_DATA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) 
          for(Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY1;
          }

        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) 
          for(Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY2;
          }

        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) 
          for(Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY3;
          }

        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) 
          for(Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY4;
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
      POLYBENCH_GEMMVER_DATA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=] (int i, int j) {
          POLYBENCH_GEMMVER_BODY1;
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=] (int i, int j) {
          POLYBENCH_GEMMVER_BODY2;
        });


        RAJA::forall<RAJA::omp_parallel_for_exec> (RAJA::RangeSegment{0, n}, [=] (int i) {
            POLYBENCH_GEMMVER_BODY3; 
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::seq_exec>>> (RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=] (int i, int j) {
          POLYBENCH_GEMMVER_BODY4;
        });
      }
      stopTimer();
#endif
      break;
    }

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {
#if 1
      POLYBENCH_GEMMVER_DATA_SETUP_CUDA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_n * m_n, block_size);
        polybench_gemmver_cuda_1<<<grid_size,block_size>>>(A,u1,v1,u2,v2,m_n);

        grid_size = RAJA_DIVIDE_CEILING_INT(m_n * m_n, block_size);
        polybench_gemmver_cuda_2<<<grid_size,block_size>>>(beta,A,x,y,m_n);

        grid_size = RAJA_DIVIDE_CEILING_INT(m_n , block_size);
        polybench_gemmver_cuda_3<<<grid_size,block_size>>>(x,z,v1,u2,v2,m_n);

        grid_size = RAJA_DIVIDE_CEILING_INT(m_n * m_n, block_size);
        polybench_gemmver_cuda_4<<<grid_size,block_size>>>(alpha,A,x,w,m_n);
      }
      cudaDeviceSynchronize();
      stopTimer();
      POLYBENCH_GEMMVER_TEARDOWN_CUDA;
#endif
      break;
    }

    case RAJA_CUDA : {
#if 1
      POLYBENCH_GEMMVER_DATA_SETUP_CUDA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
       
        RAJA::forall<RAJA::cuda_exec<block_size>> (RAJA::RangeSegment{0, n * n}, [=] __device__ (int ii) {
          Index_type i,j;
          i = ii/n; j = ii % n;
          POLYBENCH_GEMMVER_BODY1; 
        });

        RAJA::forall<RAJA::cuda_exec<block_size>> (RAJA::RangeSegment{0, n * n}, [=] __device__ (int ii) {
          Index_type i,j;
          i = ii/n; j = ii % n;
          POLYBENCH_GEMMVER_BODY2;
        });

        RAJA::forall<RAJA::cuda_exec<block_size>> (RAJA::RangeSegment{0, n}, [=] __device__ (int i) {
          POLYBENCH_GEMMVER_BODY3;
        });

        RAJA::forall<RAJA::cuda_exec<block_size>> (RAJA::RangeSegment{0, n * n}, [=] __device__ (int ii) {
          Index_type i,j;
          i = ii/n; j = ii % n;
          POLYBENCH_GEMMVER_BODY4;
        });

      }
      stopTimer();
      POLYBENCH_GEMMVER_TEARDOWN_CUDA;
#endif
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

void POLYBENCH_GEMMVER::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_w, m_n);
}

void POLYBENCH_GEMMVER::tearDown(VariantID vid)
{
  (void) vid;

}

} // end namespace basic
} // end namespace rajaperf
