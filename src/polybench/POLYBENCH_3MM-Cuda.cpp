  
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

#include "POLYBENCH_3MM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

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

__global__ void poly_3mm_1(Real_ptr E, Real_ptr A, Real_ptr B,
                           Index_type nj, Index_type nk)
{
   Index_type i = blockIdx.x;
   Index_type j = threadIdx.y;

   POLYBENCH_3MM_BODY1;

   for (Index_type k=0; k < nk; ++k) {
     POLYBENCH_3MM_BODY2;
   }
}

__global__ void poly_3mm_2(Real_ptr F, Real_ptr C, Real_ptr D,
                           Index_type nl, Index_type nm)
{
   Index_type j = blockIdx.x;
   Index_type l = threadIdx.y;

   POLYBENCH_3MM_BODY3;

   for (Index_type m=0; m < nm; ++m) {
     POLYBENCH_3MM_BODY4;
   }
}

__global__ void poly_3mm_3(Real_ptr G, Real_ptr E, Real_ptr F,
                           Index_type nl, Index_type nj)
{
   Index_type i = blockIdx.x;
   Index_type l = threadIdx.y;

   POLYBENCH_3MM_BODY5;

   for (Index_type j=0; j < nj; ++j) {
     POLYBENCH_3MM_BODY6;
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

      dim3 nblocks1(ni, 1, 1);
      dim3 nthreads_per_block1(1, nj, 1);
      poly_3mm_1<<<nblocks1, nthreads_per_block1>>>(E, A, B,
                                                    nj, nk);

      dim3 nblocks2(nj, 1, 1);
      dim3 nthreads_per_block2(1, nl, 1);
      poly_3mm_2<<<nblocks2, nthreads_per_block2>>>(F, C, D,
                                                    nl, nm);

      dim3 nblocks3(ni, 1, 1);
      dim3 nthreads_per_block3(1, nl, 1);
      poly_3mm_3<<<nblocks3, nthreads_per_block3>>>(G, E, F,
                                                    nl, nj);

    }
    stopTimer();
    
    POLYBENCH_3MM_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_3MM_DATA_SETUP_CUDA;

    POLYBENCH_3MM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelAsync<
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
          POLYBENCH_3MM_BODY1_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
          POLYBENCH_3MM_BODY2_RAJA;
        }

      );

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                                               RAJA::RangeSegment{0, nl},
                                               RAJA::RangeSegment{0, nm}),
        [=] __device__ (Index_type j, Index_type l, Index_type /* m */) {
          POLYBENCH_3MM_BODY3_RAJA;
        },
        [=] __device__ (Index_type j, Index_type l, Index_type m) {
          POLYBENCH_3MM_BODY4_RAJA;
        }

      );

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                                               RAJA::RangeSegment{0, nl},
                                               RAJA::RangeSegment{0, nj}),
        [=] __device__ (Index_type i, Index_type l, Index_type /* j */) {
          POLYBENCH_3MM_BODY5_RAJA;
        }, 
        [=] __device__ (Index_type i, Index_type l, Index_type j) {
          POLYBENCH_3MM_BODY6_RAJA;
        }
                                               
      );

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
  
