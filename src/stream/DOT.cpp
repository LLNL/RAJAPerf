/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Stream kernel DOT.
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


#include "DOT.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#define DEBUG_ME
//#undef DEBUG_ME

#include <iostream>

namespace rajaperf 
{
namespace stream
{

#define DOT_DATA \
  ResReal_ptr a = m_a; \
  ResReal_ptr b = m_b;

#define DOT_BODY  \
  dot += a[i] * b[i] ;


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define DOT_DATA_SETUP_CUDA \
  Real_ptr a; \
  Real_ptr b; \
\
  allocAndInitCudaDeviceData(a, m_a, iend); \
  allocAndInitCudaDeviceData(b, m_b, iend);

#define DOT_DATA_TEARDOWN_CUDA \
  deallocCudaDeviceData(a); \
  deallocCudaDeviceData(b);

#if 0
__global__ void dot(Real_ptr a, Real_ptr b,
                    Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
   }
}
#endif

#endif // if defined(RAJA_ENABLE_CUDA)


DOT::DOT(const RunParams& params)
  : KernelBase(rajaperf::Stream_DOT, params)
{
   setDefaultSize(1000000);
   setDefaultSamples(500);
}

DOT::~DOT() 
{
}

void DOT::setUp(VariantID vid)
{
  allocAndInitData(m_a, getRunSize(), vid);
  allocAndInitData(m_b, getRunSize(), vid);

  m_dot = 0.0;
}

void DOT::runKernel(VariantID vid)
{
  const Index_type run_samples = getRunSamples();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Baseline_Seq : {

      DOT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        Real_type dot = 0.0;

        for (Index_type i = ibegin; i < iend; ++i ) {
          DOT_BODY;
        }

         m_dot += dot;

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      DOT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::ReduceSum<RAJA::seq_reduce, Real_type> dot(0.0);

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](Index_type i) {
          DOT_BODY;
        });

        m_dot += static_cast<Real_type>(dot.get());

      }
      stopTimer();

      break;
    }

#if defined(_OPENMP)
    case Baseline_OpenMP : {

      DOT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        Real_type dot = 0.0;

        #pragma omp parallel for reduction(+:dot)
        for (Index_type i = ibegin; i < iend; ++i ) {
          DOT_BODY;
        }

        m_dot += dot;

      }
      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {
      // case is not defined...
      break;
    }

    case RAJA_OpenMP : {

      DOT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::ReduceSum<RAJA::omp_reduce, Real_type> dot(0.0);

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, 
          [=](Index_type i) {
          DOT_BODY;
        });

        m_dot += static_cast<Real_type>(dot.get());

      }
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Baseline_CUDA : {

#if 0
      DOT_DATA_SETUP_CUDA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         dot<<<grid_size, block_size>>>( a, b, 
                                         iend ); 

      }
      stopTimer();

      DOT_DATA_TEARDOWN_CUDA;
#endif

      break; 
    }

    case RAJA_CUDA : {

      DOT_DATA_SETUP_CUDA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

         RAJA::ReduceSum<RAJA::cuda_reduce<block_size>, Real_type> dot(0.0);

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend, 
           [=] __device__ (Index_type i) {
           DOT_BODY;
         });

         m_dot += static_cast<Real_type>(dot.get());

      }
      stopTimer();

      DOT_DATA_TEARDOWN_CUDA;

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

void DOT::updateChecksum(VariantID vid)
{
  checksum[vid] += m_dot;
}

void DOT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_b);
}

} // end namespace stream
} // end namespace rajaperf
