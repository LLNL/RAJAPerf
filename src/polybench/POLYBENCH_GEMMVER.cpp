//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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

///
/// POLYBENCH_GEMMVER kernel reference implementation:
///
/// for (Index_type i = 0; i < _PB_N; i++) {
///   for (Index_type j = 0; j < _PB_N; j++) {
///     A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
///   }
/// }
///
/// for (Index_type i = 0; i < _PB_N; i++) {
///   for (Index_type j = 0; j < _PB_N; j++) {
///     x[i] = x[i] + beta * A[j][i] * y[j];
///   }
/// }
///
/// for (Index_type i = 0; i < _PB_N; i++) {
///   x[i] = x[i] + z[i];
/// }
///
/// for (Index_type i = 0; i < _PB_N; i++) {
///   for (Index_type j = 0; j < _PB_N; j++) {
///     w[i] = w[i] +  alpha * A[i][j] * x[j];
///   }
/// }
///


#include "POLYBENCH_GEMMVER.hpp"

#include "common/DataUtils.hpp"
#include "common/CudaDataUtils.hpp"

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
  SizeSpec_T lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0;
  switch(lsizespec) {
    case Mini:
      m_n=40;
      run_reps = 200000;
      break;
    case Small:
      m_n=120; 
      run_reps = 20000;
      break;
    case Medium:
      m_n=400;
      run_reps = 2000;
      break;
    case Large:
      m_n=2000;
      run_reps = 20;
      break;
    case Extralarge:
      m_n=4000; 
      run_reps = 5;
      break;
    default:
      m_n=400;
      run_reps = 2000;
      break;
  }

  setDefaultSize(m_n*m_n + m_n*m_n + m_n + m_n*m_n);
  setDefaultReps(run_reps);

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
  (void) vid;
#if 0 // RDH attempt to initialize alpha and beta to non-zero values and
      // w to zero so checksum indicates whether kernel variant is run.
      // These changes break the code...
  initData(m_alpha);
  initData(m_beta);
  initDataConst(m_w, m_n, 0.0);
#endif
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

        for (Index_type i = 0; i < n; i++ ) {
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY1;
          }
        }

        for (Index_type i = 0; i < n; i++ ) { 
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY2;
          }
        }

// RDH This should not be a loop nest, only an 'i' loop
        for (Index_type i = 0; i < n; i++ ) { 
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY3;
          }
        }

        for (Index_type i = 0; i < n; i++ ) { 
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY4;
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      POLYBENCH_GEMMVER_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,
                                                        RAJA::seq_exec>>> (
          RAJA::RangeSegment{0, n}, 
          RAJA::RangeSegment{0, n}, 
          [=] (int i, int j) {
          POLYBENCH_GEMMVER_BODY1;
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,
                                                        RAJA::seq_exec>>> (
          RAJA::RangeSegment{0, n}, 
          RAJA::RangeSegment{0, n}, 
          [=] (int i, int j) {
          POLYBENCH_GEMMVER_BODY2;
        });


        RAJA::forall<RAJA::seq_exec> (
          RAJA::RangeSegment{0, n}, [=] (int i) {
          POLYBENCH_GEMMVER_BODY3; 
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,
                                                        RAJA::seq_exec>>> (
          RAJA::RangeSegment{0, n}, 
          RAJA::RangeSegment{0, n}, 
          [=] (int i, int j) {
          POLYBENCH_GEMMVER_BODY4;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)      
    case Base_OpenMP : {

      POLYBENCH_GEMMVER_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) {
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY1;
          }
        }

        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) {
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY2;
          }
        } 

        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) {
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY3;
          }
        }

        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) {
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY4;
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_GEMMVER_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,
                                                        RAJA::seq_exec>>> (
          RAJA::RangeSegment{0, n}, 
          RAJA::RangeSegment{0, n}, 
          [=] (int i, int j) {
          POLYBENCH_GEMMVER_BODY1;
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,
                                                        RAJA::seq_exec>>> (
          RAJA::RangeSegment{0, n}, 
          RAJA::RangeSegment{0, n}, 
          [=] (int i, int j) {
          POLYBENCH_GEMMVER_BODY2;
        });


        RAJA::forall<RAJA::omp_parallel_for_exec> (
          RAJA::RangeSegment{0, n}, [=] (int i) {
          POLYBENCH_GEMMVER_BODY3; 
        });

        RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,
                                                        RAJA::seq_exec>>> (
          RAJA::RangeSegment{0, n}, 
          RAJA::RangeSegment{0, n}, 
          [=] (int i, int j) {
          POLYBENCH_GEMMVER_BODY4;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#define NUMTEAMS 128

    case Base_OpenMPTarget : {

      POLYBENCH_GEMMVER_DATA;

      #pragma omp target enter data map(to: A[0:n*n],u1[0:n], v1[0:n], u2[0:n], v2[0: n], w[0:n], x[0:n], y[0:n], z[0:n], alpha, beta)

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp target teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) collapse(2)
        
        for (Index_type i = 0; i < n; i++ ) {
          for(Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY1;
          }
        }

        #pragma omp target teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) collapse(2)
        for (Index_type i = 0; i < n; i++ ) { 
          for(Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY2;
          }
        }

        #pragma omp target teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) collapse(2)
        for (Index_type i = 0; i < n; i++ ) {
          for(Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY3;
          }
        }

        #pragma omp target teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) collapse(2)
        for (Index_type i = 0; i < n; i++ ) {
          for(Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY4;
          }
        }

      }
      stopTimer();

      #pragma omp target exit data map(from:w[0:m_n]) map(delete: u1[0:m_n],v1[0:m_n], u2[0:m_n], v2[0:m_n], x[0: m_n], y[0: m_n], z[0:m_n], alpha, beta)

      break;
    }

    case RAJA_OpenMPTarget: {

      POLYBENCH_GEMMVER_DATA;

      #pragma omp target enter data map(to: A[0:n*n],u1[0:n], v1[0:n], u2[0:n], v2[0: n], w[0:n], x[0:n], y[0:n], z[0:n], alpha, beta)

      startTimer();
      #pragma omp target data use_device_ptr(A,u1,v1,u2,v2,w,x,y,z)
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::policy::omp::omp_target_parallel_for_exec<NUMTEAMS>>(
            RAJA::RangeSegment(0,n * n), [=](Index_type ii) {
          Index_type i,j;
          i = ii/n; j = ii % n;
          POLYBENCH_GEMMVER_BODY1; 
        });

        RAJA::forall<RAJA::policy::omp::omp_target_parallel_for_exec<NUMTEAMS>>(
            RAJA::RangeSegment(0,n * n), [=](Index_type ii) {
          Index_type i,j;
          i = ii/n; j = ii % n;
          POLYBENCH_GEMMVER_BODY2; 
        });

        RAJA::forall<RAJA::policy::omp::omp_target_parallel_for_exec<NUMTEAMS>>(
            RAJA::RangeSegment(0,n), [=](Index_type i) {
          POLYBENCH_GEMMVER_BODY3; 
        });

        RAJA::forall<RAJA::policy::omp::omp_target_parallel_for_exec<NUMTEAMS>>(
            RAJA::RangeSegment(0,n * n), [=](Index_type ii) {
          Index_type i,j;
          i = ii/n; j = ii % n;
          POLYBENCH_GEMMVER_BODY4; 
        });

      } // for run_reps   
      stopTimer();

      #pragma omp target exit data map(from:w[0:m_n]) map(delete: u1[0:m_n],v1[0:m_n], u2[0:m_n], v2[0:m_n], x[0: m_n], y[0: m_n], z[0:m_n], alpha, beta)
    
      break;                        
    }  
#endif //RAJA_ENABLE_TARGET_OPENMP
#endif //RAJA_ENABLE_OMP                             
     

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {

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

      break;
    }

    case RAJA_CUDA : {
      POLYBENCH_GEMMVER_DATA_SETUP_CUDA;
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
       
        RAJA::forall<RAJA::cuda_exec<block_size>> (
          RAJA::RangeSegment{0, n * n}, [=] __device__ (int ii) {
          Index_type i,j;
          i = ii/n; j = ii % n;
          POLYBENCH_GEMMVER_BODY1; 
        });

        RAJA::forall<RAJA::cuda_exec<block_size>> (
          RAJA::RangeSegment{0, n * n}, [=] __device__ (int ii) {
          Index_type i,j;
          i = ii/n; j = ii % n;
          POLYBENCH_GEMMVER_BODY2;
        });

        RAJA::forall<RAJA::cuda_exec<block_size>> (
          RAJA::RangeSegment{0, n}, [=] __device__ (int i) {
          POLYBENCH_GEMMVER_BODY3;
        });

        RAJA::forall<RAJA::cuda_exec<block_size>> (
          RAJA::RangeSegment{0, n * n}, [=] __device__ (int ii) {
          Index_type i,j;
          i = ii/n; j = ii % n;
          POLYBENCH_GEMMVER_BODY4;
        });

      }
      stopTimer();
      POLYBENCH_GEMMVER_TEARDOWN_CUDA;
      break;
    }

#endif

    default : {
      std::cout << "\n  POLYBENCH_GEMMVER : Unknown variant id = " << vid << std::endl;
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
