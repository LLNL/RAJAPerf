  
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
// For details about use and distribution, please read RAJAPerf/LICENSE.
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
  const Index_type n = m_n; \
  const Index_type tsteps = m_tsteps; \
\
  Real_type DX,DY,DT; \
  Real_type B1,B2; \
  Real_type mul1,mul2; \
  Real_type a,b,c,d,e,f; \
\
  Real_ptr U; \
  Real_ptr V; \
  Real_ptr P; \
  Real_ptr Q; \
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


__global__ void adi1(const Index_type n,
                     const Real_type a, const Real_type b, const Real_type c, 
                     const Real_type d, const Real_type f,
                     Real_ptr P, Real_ptr Q, Real_ptr U, Real_ptr V)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > 0 && i < n-1) {
    NEW_POLYBENCH_ADI_BODY2;
    for (Index_type j = 1; j < n-1; ++j) {
       NEW_POLYBENCH_ADI_BODY3;
    }
    NEW_POLYBENCH_ADI_BODY4;
    for (Index_type k = n-2; k >= 1; --k) {
       NEW_POLYBENCH_ADI_BODY5;
    }
  }
}

__global__ void adi2(const Index_type n,
                     const Real_type a, const Real_type c, const Real_type d, 
                     const Real_type e, const Real_type f,
                     Real_ptr P, Real_ptr Q, Real_ptr U, Real_ptr V)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > 0 && i < n-1) {
    NEW_POLYBENCH_ADI_BODY6;
    for (Index_type j = 1; j < n-1; ++j) {
      NEW_POLYBENCH_ADI_BODY7;
    }
    NEW_POLYBENCH_ADI_BODY8;
    for (Index_type k = n-2; k >= 1; --k) {
      NEW_POLYBENCH_ADI_BODY9;
    }
  }
}


void POLYBENCH_ADI::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  
  if ( vid == Base_CUDA ) {

    POLYBENCH_ADI_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 1; t <= tsteps; ++t) {

        POLYBENCH_ADI_BODY1;

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(n-1, block_size);

        adi1<<<grid_size, block_size>>>(n,
                                        a, b, c, d, f,
                                        P, Q, U, V);

        adi2<<<grid_size, block_size>>>(n,
                                        a, c, d, e, f,
                                        P, Q, U, V);

      }  // tstep loop

    }
    stopTimer();

    POLYBENCH_ADI_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {   

    POLYBENCH_ADI_DATA_SETUP_CUDA

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
          RAJA::statement::For<0, RAJA::cuda_threadblock_exec<block_size>,
            RAJA::statement::Lambda<0>,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<1>
            >,
            RAJA::statement::Lambda<2>,
            RAJA::statement::For<2, RAJA::seq_exec,
              RAJA::statement::Lambda<3>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      POLYBENCH_ADI_BODY1;

      for (Index_type t = 1; t <= tsteps; ++t) {

        RAJA::kernel<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                           RAJA::RangeSegment{1, n-1},
                           RAJA::RangeStrideSegment{n-2, 0, -1}),

          [=] __device__ (Index_type i, Index_type /*j*/, Index_type /*k*/) {
            NEW_POLYBENCH_ADI_BODY2;
          },
          [=] __device__ (Index_type i, Index_type j, Index_type /*k*/) {
            NEW_POLYBENCH_ADI_BODY3;
          },
          [=] __device__ (Index_type i, Index_type /*j*/, Index_type /*k*/) {
            NEW_POLYBENCH_ADI_BODY4;
          },
          [=] __device__ (Index_type i, Index_type /*j*/, Index_type k) {
            NEW_POLYBENCH_ADI_BODY5;
          }
        );

        RAJA::kernel<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                           RAJA::RangeSegment{1, n-1},
                           RAJA::RangeStrideSegment{n-2, 0, -1}),

          [=] __device__ (Index_type i, Index_type /*j*/, Index_type /*k*/) {
            NEW_POLYBENCH_ADI_BODY6;
          },
          [=] __device__ (Index_type i, Index_type j, Index_type /*k*/) {
            NEW_POLYBENCH_ADI_BODY7;
          },
          [=] __device__ (Index_type i, Index_type /*j*/, Index_type /*k*/) {
            NEW_POLYBENCH_ADI_BODY8;
          },
          [=] __device__ (Index_type i, Index_type /*j*/, Index_type k) {
            NEW_POLYBENCH_ADI_BODY9;
          }
        );

      }  // tstep loop

    } // run_reps
    stopTimer();

    POLYBENCH_ADI_TEARDOWN_CUDA

  } else {
      std::cout << "\n  POLYBENCH_ADI : Unknown Cuda variant id = " << vid << std::endl;
  }
}
  
} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
  
