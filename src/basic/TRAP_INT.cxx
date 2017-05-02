/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Basic kernel TRAP_INT.
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


#include "TRAP_INT.hxx"

#include "common/DataUtils.hxx"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define TRAP_INT_DATA 

#define TRAP_INT_BODY 

#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define TRAP_INT_DATA_SETUP_CUDA

#define TRAP_INT_DATA_TEARDOWN_CUDA

__global__ void trapint(Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     TRAP_INT_BODY;
   }
}

#endif // if defined(RAJA_ENABLE_CUDA)


TRAP_INT::TRAP_INT(const RunParams& params)
  : KernelBase(rajaperf::Basic_TRAP_INT, params)
{
   setDefaultSize(100000);
   setDefaultSamples(1500);
}

TRAP_INT::~TRAP_INT() 
{
}

void TRAP_INT::setUp(VariantID vid)
{
  (void) vid;
}

void TRAP_INT::runKernel(VariantID vid)
{
  const Index_type run_samples = getRunSamples();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Baseline_Seq : {

      TRAP_INT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          TRAP_INT_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      TRAP_INT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          TRAP_INT_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(_OPENMP)
    case Baseline_OpenMP : {

      TRAP_INT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        #pragma omp for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          TRAP_INT_BODY;
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

      TRAP_INT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, 
          [=](Index_type i) {
          TRAP_INT_BODY;
        });


      }
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Baseline_CUDA : {

      TRAP_INT_DATA_SETUP_CUDA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         trapint<<<grid_size, block_size>>>( iend );

      }
      stopTimer();

      TRAP_INT_DATA_TEARDOWN_CUDA;

      break;
    }

    case RAJA_CUDA : {

      TRAP_INT_DATA_SETUP_CUDA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

         RAJA::forall< RAJA::cuda_exec<block_size> >(ibegin, iend,
           [=] __device__ (Index_type i) {
           TRAP_INT_BODY;
         });

      }
      stopTimer();

      TRAP_INT_DATA_TEARDOWN_CUDA;

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

void TRAP_INT::updateChecksum(VariantID vid)
{
  (void) vid;
}

void TRAP_INT::tearDown(VariantID vid)
{
  (void) vid;
}

} // end namespace basic
} // end namespace rajaperf
