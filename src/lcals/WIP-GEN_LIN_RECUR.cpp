/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Lcals kernel GEN_LIN_RECUR.
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
// For more information, please see the file LICENSE in the top-level directory.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "GEN_LIN_RECUR.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

#define GEN_LIN_RECUR_DATA \
  ResReal_ptr b5 = m_b5; \
  ResReal_ptr sa = m_sa; \
  ResReal_ptr sb = m_sb; \
  Real_type stb5 = m_stb5; \
  const Index_type kb5i = m_kb5i; \
  const Index_type len = m_len;

//
// How to parallelize these? Scans?
//

#define GEN_LIN_RECUR_BODY1  \
  b5[k+kb5i] = sa[k] + stb5*sb[k]; \
        stb5 = b5[k+kb5i] - stb5;

#define GEN_LIN_RECUR_BODY2  \
  Index_type k = len - i ; \
    b5[k+kb5i] = sa[k] + stb5*sb[k]; \
          stb5 = b5[k+kb5i] - stb5;


#if defined(ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define GEN_LIN_RECUR_DATA_SETUP_CUDA \
  Real_ptr b5; \
  Real_ptr sa; \
  Real_ptr sb; \
  Real_type stb5 = m_stb5; \
  const Index_type kb5i = m_kb5i; \
  const Index_type len = m_len; \
\
  allocAndInitCudaDeviceData(b5, m_b5, len+kb5i+1); \
  allocAndInitCudaDeviceData(sa, m_sa, len+kb5i+1); \
  allocAndInitCudaDeviceData(sb, m_sb, len+kb5i+1);

#define GEN_LIN_RECUR_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_b5, b5, len+kb5i+1); \
  deallocCudaDeviceData(b5); \
  deallocCudaDeviceData(sa); \
  deallocCudaDeviceData(sb);

__global__ void gen_lin_recur1(Real_ptr b5, Real_ptr sa, Real_ptr sb,
                               Real_type stb5, const Index_type kb5i,
                               Index_type kbegin,
                               Index_type kend) 
{
   Index_type k = kbegin + blockIdx.x * blockDim.x + threadIdx.x;
   if (k < kend) {
     GEN_LIN_RECUR_BODY1; 
   }
}

__global__ void gen_lin_recur2(Real_ptr b5, Real_ptr sa, Real_ptr sb,
                               Real_type stb5, const Index_type kb5i,
                               const Index_type len,
                               Index_type ibegin,
                               Index_type iend) 
{
   Index_type i = ibegin + blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     GEN_LIN_RECUR_BODY2;
   }
}

#endif // if defined(ENABLE_CUDA)


GEN_LIN_RECUR::GEN_LIN_RECUR(const RunParams& params)
  : KernelBase(rajaperf::Lcals_GEN_LIN_RECUR, params)
{
   setDefaultSize(100000);
   setDefaultReps(2000);
}

GEN_LIN_RECUR::~GEN_LIN_RECUR() 
{
}

void GEN_LIN_RECUR::setUp(VariantID vid)
{
  m_len = getRunSize();

  m_kb5i = 0;
 
  allocAndInitData(m_b5, m_len+m_kb5i+1, vid);
  allocAndInitData(m_sa, m_len+m_kb5i+1, vid);
  allocAndInitData(m_sb, m_len+m_kb5i+1, vid);

  initData(m_stb5, vid);
}

void GEN_LIN_RECUR::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  switch ( vid ) {

    case Base_Seq : {

      GEN_LIN_RECUR_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < len; ++k ) {
          GEN_LIN_RECUR_BODY1;
        }

        for (Index_type i = 1; i < len+1; ++i ) {
          GEN_LIN_RECUR_BODY2;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      GEN_LIN_RECUR_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(0, len, [=](Index_type k) {
          GEN_LIN_RECUR_BODY1;
        });

        RAJA::forall<RAJA::simd_exec>(1, len+1, [=](Index_type i) {
          GEN_LIN_RECUR_BODY2;
        });

      }
      stopTimer();

      break;
    }

#if defined(ENABLE_OPENMP)
    case Base_OpenMP : {

      GEN_LIN_RECUR_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type k = 0; k < len; ++k ) {
          GEN_LIN_RECUR_BODY1;
        }

        #pragma omp parallel for
        for (Index_type i = 1; i < len+1; ++i ) {
          GEN_LIN_RECUR_BODY2;
        }

      }
      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {
      // case is not defined...
      break;
    }

    case RAJA_OpenMP : {

      GEN_LIN_RECUR_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(0, len, [=](Index_type k) {
          GEN_LIN_RECUR_BODY1;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(1, len+1, [=](Index_type i) {
          GEN_LIN_RECUR_BODY2;
        });

      }
      stopTimer();

      break;
    }
#endif

#if defined(ENABLE_CUDA)
    case Base_CUDA : {

      GEN_LIN_RECUR_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(len, block_size);

         gen_lin_recur1<<<grid_size, block_size>>>(b5, sa, sb,
                                                   stb5, kb5i,
                                                   0, len);

         gen_lin_recur2<<<grid_size, block_size>>>(b5, sa, sb,
                                                   stb5, kb5i,
                                                   len,
                                                   1, len+1);

      }
      stopTimer();

      GEN_LIN_RECUR_DATA_TEARDOWN_CUDA;

      break; 
    }

    case RAJA_CUDA : {

      GEN_LIN_RECUR_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           0, len, 
           [=] __device__ (Index_type k) {
           GEN_LIN_RECUR_BODY1;
         });

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           1, len+1, 
           [=] __device__ (Index_type i) {
           GEN_LIN_RECUR_BODY2;
         });

      }
      stopTimer();

      GEN_LIN_RECUR_DATA_TEARDOWN_CUDA;

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

void GEN_LIN_RECUR::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_b5, getRunSize());
}

void GEN_LIN_RECUR::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_b5);
  deallocData(m_sa);
  deallocData(m_sb);
}

} // end namespace lcals
} // end namespace rajaperf
