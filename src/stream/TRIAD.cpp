/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Stream kernel TRIAD.
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


#include "TRIAD.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{

#define TRIAD_DATA \
  ResReal_ptr a = m_a; \
  ResReal_ptr b = m_b; \
  ResReal_ptr c = m_c; \
  Real_type alpha = m_alpha;

#define TRIAD_BODY  \
  a[i] = b[i] + alpha * c[i] ;


#if defined(ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define TRIAD_DATA_SETUP_CUDA \
  Real_ptr a; \
  Real_ptr b; \
  Real_ptr c; \
  Real_type alpha = m_alpha; \
\
  allocAndInitCudaDeviceData(a, m_a, iend); \
  allocAndInitCudaDeviceData(b, m_b, iend); \
  allocAndInitCudaDeviceData(c, m_c, iend);

#define TRIAD_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_a, a, iend); \
  deallocCudaDeviceData(a); \
  deallocCudaDeviceData(b); \
  deallocCudaDeviceData(c)

__global__ void triad(Real_ptr a, Real_ptr b, Real_ptr c, Real_type alpha,
                      Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     TRIAD_BODY; 
   }
}

#endif // if defined(ENABLE_CUDA)


TRIAD::TRIAD(const RunParams& params)
  : KernelBase(rajaperf::Stream_TRIAD, params)
{
   setDefaultSize(1000000);
   setDefaultReps(800);
}

TRIAD::~TRIAD() 
{
}

void TRIAD::setUp(VariantID vid)
{
  allocAndInitData(m_a, getRunSize(), vid);
  allocAndInitData(m_b, getRunSize(), vid);
  allocAndInitData(m_c, getRunSize(), vid);
  initData(m_alpha, vid);
}

void TRIAD::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      TRIAD_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          TRIAD_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      TRIAD_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](Index_type i) {
          TRIAD_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(ENABLE_OPENMP)
    case Base_OpenMP : {

      TRIAD_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          TRIAD_BODY;
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

      TRIAD_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, 
          [=](Index_type i) {
          TRIAD_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif

#if defined(ENABLE_CUDA)
    case Base_CUDA : {

      TRIAD_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         triad<<<grid_size, block_size>>>( a, b, c, alpha,
                                           iend ); 

      }
      stopTimer();

      TRIAD_DATA_TEARDOWN_CUDA;

      break; 
    }

    case RAJA_CUDA : {

      TRIAD_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend, 
           [=] __device__ (Index_type i) {
           TRIAD_BODY;
         });

      }
      stopTimer();

      TRIAD_DATA_TEARDOWN_CUDA;

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

void TRIAD::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_a, getRunSize());
}

void TRIAD::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_b);
  deallocData(m_c);
}

} // end namespace stream
} // end namespace rajaperf
