/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Basic kernel MULADDSUB.
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


#include "MULADDSUB.hxx"

#include "common/DataUtils.hxx"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define MULADDSUB_DATA \
  ResReal_ptr out1 = m_out1; \
  ResReal_ptr out2 = m_out2; \
  ResReal_ptr out3 = m_out3; \
  ResReal_ptr in1 = m_in1; \
  ResReal_ptr in2 = m_in2;


#define MULADDSUB_BODY  \
  out1[i] = in1[i] * in2[i] ; \
  out2[i] = in1[i] + in2[i] ; \
  out3[i] = in1[i] - in2[i] ;


MULADDSUB::MULADDSUB(const RunParams& params)
  : KernelBase(rajaperf::Basic_MULADDSUB, params)
{
   setDefaultSize(100000);
   setDefaultSamples(6000);
}

MULADDSUB::~MULADDSUB() 
{
}

void MULADDSUB::setUp(VariantID vid)
{
  allocAndInit(m_out1, getRunSize(), vid);
  allocAndInit(m_out2, getRunSize(), vid);
  allocAndInit(m_out3, getRunSize(), vid);
  allocAndInit(m_in1, getRunSize(), vid);
  allocAndInit(m_in2, getRunSize(), vid);
}

void MULADDSUB::runKernel(VariantID vid)
{
  const Index_type run_samples = getRunSamples();
  const Index_type lbegin = 0;
  const Index_type lend = getRunSize();

  switch ( vid ) {

    case Baseline_Seq : {

      MULADDSUB_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        for (Index_type i = lbegin; i < lend; ++i ) {
          MULADDSUB_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      MULADDSUB_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::simd_exec>(lbegin, lend, [=](int i) {
          MULADDSUB_BODY;
        });

      }
      stopTimer();

      break;
    }

    case Baseline_OpenMP : {
#if defined(_OPENMP)

      MULADDSUB_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        #pragma omp parallel for
        for (Index_type i = lbegin; i < lend; ++i ) {
          MULADDSUB_BODY;
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

      MULADDSUB_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(lbegin, lend, [=](int i) {
          MULADDSUB_BODY;
        });

      }
      stopTimer();

      break;
    }

    case Baseline_CUDA : {
#if defined(RAJA_ENABLE_CUDA) || 0

      MULADDSUB_DATA;

      cudaMalloc( (void**)&out1, getRunSize() * sizeof(Real_type) );
      cudaMalloc( (void**)&out2, getRunSize() * sizeof(Real_type) );
      cudaMalloc( (void**)&out3, getRunSize() * sizeof(Real_type) );
      cudaMalloc( (void**)&in1, getRunSize() * sizeof(Real_type) );
      cudaMalloc( (void**)&in2, getRunSize() * sizeof(Real_type) );

      cudaMemcpy( out1, m_out1, getRunSize() * sizeof(Real_type),
                  cudaMemcpyHostToDevice );
      cudaMemcpy( out2, m_out2, getRunSize() * sizeof(Real_type),
                  cudaMemcpyHostToDevice );
      cudaMemcpy( out3, m_out3, getRunSize() * sizeof(Real_type),
                  cudaMemcpyHostToDevice );
      cudaMemcpy( in1, m_in1, getRunSize() * sizeof(Real_type),
                  cudaMemcpyHostToDevice );
      cudaMemcpy( in2, m_in2, getRunSize() * sizeof(Real_type),
                  cudaMemcpyHostToDevice );

/*
__global__ void muladdsub(Real_ptr out1, Real_ptr out2, Real_ptr out3, 
                          Real_prt in1, Real_ptr in2, Index_type lend) 
{
   int tid i = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid < lend) {
     MULADDSUB_BODY; 
   }
}

#define RAJA_DIVIDE_CEILING_INT(dividend, divisor) \
 ( ( (dividend) + (divisor) - 1 ) / (divisor) )
*/

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

         Index_type grid_size = RAJA_DIVIDE_CEILING_INT(len, BLOCK_SIZE);

         muladdsub<<<grid_size, block_size>>>( out1, out2, out3, in1, in2 ); 

      }
      stopTimer();

      cudaMemcpy( m_out1, out1, getRunSize() * sizeof(Real_type),
                  cudaMemcpyDeviceToHost );
      cudaMemcpy( m_out2, out2, getRunSize() * sizeof(Real_type),
                  cudaMemcpyDeviceToHost );
      cudaMemcpy( m_out3, out3, getRunSize() * sizeof(Real_type),
                  cudaMemcpyDeviceToHost );

#endif
      break; 
    }

    case RAJA_CUDA : {
#if defined(RAJA_ENABLE_CUDA)
      // Fill these in later...you get the idea...
#endif
      break;
    }

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

void MULADDSUB::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_out1, getRunSize());
  checksum[vid] += calcChecksum(m_out2, getRunSize());
  checksum[vid] += calcChecksum(m_out3, getRunSize());
}

void MULADDSUB::tearDown(VariantID vid)
{
  dealloc(m_out1);
  dealloc(m_out2);
  dealloc(m_out3);
  dealloc(m_in1);
  dealloc(m_in2);

  if (vid == Baseline_CUDA || vid == RAJA_CUDA) {
    // De-allocate device memory here.
  }
}

} // end namespace basic
} // end namespace rajaperf
