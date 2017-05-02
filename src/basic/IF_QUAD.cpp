/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Basic kernel IF_QUAD.
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


#include "IF_QUAD.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define IF_QUAD_DATA \
  ResReal_ptr a = m_a; \
  ResReal_ptr b = m_b; \
  ResReal_ptr c = m_c; \
  ResReal_ptr x1 = m_x1; \
  ResReal_ptr x2 = m_x2;

#define IF_QUAD_BODY  \
  Real_type s = b[i]*b[i] - 4.0*a[i]*c[i]; \
  if ( s >= 0 ) { \
    s = sqrt(s); \
    x2[i] = (-b[i]+s)/(2.0*a[i]); \
    x1[i] = (-b[i]-s)/(2.0*a[i]); \
  } else { \
    x2[i] = 0.0; \
    x1[i] = 0.0; \
  }

#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define IF_QUAD_DATA_SETUP_CUDA \
  Real_ptr a; \
  Real_ptr b; \
  Real_ptr c; \
  Real_ptr x1; \
  Real_ptr x2; \
\
  allocAndInitCudaDeviceData(a, m_a, iend); \
  allocAndInitCudaDeviceData(b, m_b, iend); \
  allocAndInitCudaDeviceData(c, m_c, iend); \
  allocAndInitCudaDeviceData(x1, m_x1, iend); \
  allocAndInitCudaDeviceData(x2, m_x2, iend);

#define MULADDSUB_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x1, x1, iend); \
  getCudaDeviceData(m_x2, x2, iend); \
  deallocCudaDeviceData(x1); \
  deallocCudaDeviceData(x2); \
  deallocCudaDeviceData(a); \
  deallocCudaDeviceData(b); \
  deallocCudaDeviceData(c);

__global__ void ifquad(Real_ptr x1, Real_ptr x2,
                       Real_ptr a, Real_ptr b, Real_ptr c,
                       Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     IF_QUAD_BODY;
   }
}

#endif // if defined(RAJA_ENABLE_CUDA)


IF_QUAD::IF_QUAD(const RunParams& params)
  : KernelBase(rajaperf::Basic_IF_QUAD, params)
{
   setDefaultSize(100000);
   setDefaultSamples(1500);
}

IF_QUAD::~IF_QUAD() 
{
}

void IF_QUAD::setUp(VariantID vid)
{
  allocAndInitData(m_a, getRunSize(), vid);
  allocAndInitData(m_b, getRunSize(), vid);
  allocAndInitData(m_c, getRunSize(), vid);
  allocAndInitData(m_x1, getRunSize(), vid);
  allocAndInitData(m_x2, getRunSize(), vid);
}

void IF_QUAD::runKernel(VariantID vid)
{
  const Index_type run_samples = getRunSamples();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Baseline_Seq : {

      IF_QUAD_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          IF_QUAD_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      IF_QUAD_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          IF_QUAD_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(_OPENMP)
    case Baseline_OpenMP : {

      IF_QUAD_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        #pragma omp for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          IF_QUAD_BODY;
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

      IF_QUAD_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          IF_QUAD_BODY;
        });


      }
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Baseline_CUDA : {

      IF_QUAD_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         muladdsub<<<grid_size, block_size>>>( x1, x2, a, b, c,
                                               iend );

      }
      stopTimer();

      break;
    }

    case RAJA_CUDA : {

      IF_QUAD_DATA_SETUP_CUDA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

         RAJA::forall< RAJA::cuda_exec<block_size> >(ibegin, iend,
           [=] __device__ (Index_type i) {
           IF_QUAD_BODY;
         });

      }
      stopTimer();

      IF_QUAD_DATA_TEARDOWN_CUDA;

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

void IF_QUAD::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x1, getRunSize());
  checksum[vid] += calcChecksum(m_x2, getRunSize());
}

void IF_QUAD::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_b);
  deallocData(m_c);
  deallocData(m_x1);
  deallocData(m_x2);
}

} // end namespace basic
} // end namespace rajaperf
