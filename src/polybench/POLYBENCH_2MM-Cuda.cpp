  
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

#define POLYBENCH_2MM_DATA_SETUP_CUDA \
  Real_ptr tmp = m_tmp; \
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  Real_ptr C = m_C; \
  Real_ptr D = m_D; \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
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


__global__ void poly_2mm_1(Real_ptr tmp, Real_ptr A, Real_ptr B,
                           Real_type alpha,
                           Index_type nj, Index_type nk)
{
   Index_type i = blockIdx.x;
   Index_type j = threadIdx.y;

   POLYBENCH_2MM_BODY1;

   for (Index_type k=0; k < nk; ++k) {
     POLYBENCH_2MM_BODY2;              
   }
}

__global__ void poly_2mm_2(Real_ptr tmp, Real_ptr C, Real_ptr D,
                           Real_type beta,
                           Index_type nl, Index_type nj)
{
   Index_type i = blockIdx.x;
   Index_type l = threadIdx.y;

   POLYBENCH_2MM_BODY3;

   for (Index_type j=0; j < nj; ++j) {
     POLYBENCH_2MM_BODY4;
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

      dim3 nblocks1(ni, 1, 1);
      dim3 nthreads_per_block1(1, nj, 1);
      poly_2mm_1<<<nblocks1, nthreads_per_block1>>>(tmp, A, B, alpha,
                                                    m_nj, m_nk);

      if ( irep == run_reps - 1 ) {
        cudaErrchk( cudaMemsetAsync(D, 0, m_ni * m_nl * sizeof(Real_type)) );
      }

      dim3 nblocks2(ni, 1, 1);
      dim3 nthreads_per_block2(1, nl, 1);
      poly_2mm_2<<<nblocks2, nthreads_per_block2>>>(tmp, C, D, beta,
                                                    m_nl, m_nj);

    }
    stopTimer();

    POLYBENCH_2MM_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_2MM_DATA_SETUP_CUDA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernel<
          RAJA::statement::For<0, RAJA::cuda_block_exec,
            RAJA::statement::For<1, RAJA::cuda_thread_exec,
              RAJA::statement::Lambda<0>,
              RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<1>
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                                               RAJA::RangeSegment{0, nj},
                                               RAJA::RangeSegment{0, nk}),
        [=] __device__ (Index_type i, Index_type j, Index_type /* k */) {
          POLYBENCH_2MM_BODY1;
        },
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
          POLYBENCH_2MM_BODY2;
        }
      );

      if ( irep == run_reps - 1 ) {
        cudaErrchk( cudaMemsetAsync(D, 0, m_ni * m_nl * sizeof(Real_type)) );
      }

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                                               RAJA::RangeSegment{0, nl},
                                               RAJA::RangeSegment{0, nj}),
        [=] __device__ (Index_type i, Index_type l, Index_type /* j */) {
          POLYBENCH_2MM_BODY3;
        },
        [=] __device__ (Index_type i, Index_type l, Index_type j) {
          POLYBENCH_2MM_BODY4;
        }
      );

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
  
