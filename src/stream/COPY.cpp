/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Stream kernel COPY.
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


#include "COPY.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{

#define COPY_DATA \
  ResReal_ptr a = m_a; \
  ResReal_ptr c = m_c;

#define COPY_BODY  \
  c[i] = a[i] ;


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define COPY_DATA_SETUP_CUDA \
  Real_ptr a; \
  Real_ptr c; \
\
  allocAndInitCudaDeviceData(a, m_a, iend); \
  allocAndInitCudaDeviceData(c, m_c, iend);

#define COPY_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_c, c, iend); \
  deallocCudaDeviceData(a); \
  deallocCudaDeviceData(c)

__global__ void copy(Real_ptr c, Real_ptr a,
                     Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     COPY_BODY; 
   }
}

#endif // if defined(RAJA_ENABLE_CUDA)


COPY::COPY(const RunParams& params)
  : KernelBase(rajaperf::Stream_COPY, params)
{
   setDefaultSize(1000000);
   setDefaultSamples(500);
}

COPY::~COPY() 
{
}

void COPY::setUp(VariantID vid)
{
  allocAndInitData(m_a, getRunSize(), vid);
  allocAndInitData(m_c, getRunSize(), vid);
}

void COPY::runKernel(VariantID vid)
{
  const Index_type run_samples = getRunSamples();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Baseline_Seq : {

      COPY_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          COPY_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      COPY_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](Index_type i) {
          COPY_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(_OPENMP)
    case Baseline_OpenMP : {

      COPY_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          COPY_BODY;
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

      COPY_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, 
          [=](Index_type i) {
          COPY_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Baseline_CUDA : {

      COPY_DATA_SETUP_CUDA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         copy<<<grid_size, block_size>>>( c, a,
                                          iend ); 

      }
      stopTimer();

      COPY_DATA_TEARDOWN_CUDA;

      break; 
    }

    case RAJA_CUDA : {

      COPY_DATA_SETUP_CUDA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend, 
           [=] __device__ (Index_type i) {
           COPY_BODY;
         });

      }
      stopTimer();

      COPY_DATA_TEARDOWN_CUDA;

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

void COPY::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_c, getRunSize());
}

void COPY::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_c);
}

} // end namespace stream
} // end namespace rajaperf
