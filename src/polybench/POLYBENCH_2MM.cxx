/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Polybench kernel 2MM
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


#include "POLYBENCH_2MM.hxx"

#include "common/DataUtils.hxx"

#include <iostream>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_2MM_DATA 

#define POLYBENCH_2MM_BODY


POLYBENCH_2MM::POLYBENCH_2MM(const RunParams& params)
  : KernelBase(rajaperf::Polybench_2MM, params)
{
   setDefaultSize(100000);
   setDefaultSamples(10000);
}

POLYBENCH_2MM::~POLYBENCH_2MM() 
{
}

void POLYBENCH_2MM::setUp(VariantID vid)
{
  (void) vid;
}

void POLYBENCH_2MM::runKernel(VariantID vid)
{
#if 0
  Index_type run_size = getRunSize();
#endif
  const Index_type run_samples = getRunSamples();

  switch ( vid ) {

    case Baseline_Seq : {

      POLYBENCH_2MM_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
        for (Index_type i = 0; i < run_size; ++i ) {
          POLYBENCH_2MM_BODY;
        }
#endif

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      POLYBENCH_2MM_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
        RAJA::forall<RAJA::simd_exec>(0, run_size, [=](int i) {
          POLYBENCH_2MM_BODY;
        });
#endif

      }
      stopTimer();

      break;
    }

    case Baseline_OpenMP : {

      POLYBENCH_2MM_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
        #pragma omp for schedule(static)
        for (Index_type i = 0; i < run_size; ++i ) {
          POLYBENCH_2MM_BODY;
        }
#endif

      }
      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {
      // case is not defined...
      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_2MM_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
        RAJA::forall<RAJA::omp_parallel_for_exec>(0, run_size, [=](int i) {
          POLYBENCH_2MM_BODY;
        });
#endif

      }
      stopTimer();

      break;
    }

    case Baseline_CUDA : {

      POLYBENCH_2MM_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
#endif
      }
      stopTimer();

      break;
    }

    case RAJA_CUDA : {

      POLYBENCH_2MM_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
#endif
      }
      stopTimer();

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

void POLYBENCH_2MM::updateChecksum(VariantID vid)
{
  (void) vid;
#if 0
  checksum[vid] += calcChecksum(m_p_new, getRunSize());
#endif
}

void POLYBENCH_2MM::tearDown(VariantID vid)
{
  (void) vid;
}

} // end namespace basic
} // end namespace rajaperf
