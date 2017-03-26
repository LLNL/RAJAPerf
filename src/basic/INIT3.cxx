/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Basic kernel INIT3.
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


#include "INIT3.hxx"

#include "common/DataUtils.hxx"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define INIT3_DATA 

#define INIT3_BODY(i) 


INIT3::INIT3(const RunParams& params)
  : KernelBase(rajaperf::Basic_INIT3, params)
{
   setDefaultSize(100000);
   setDefaultSamples(10000);
}

INIT3::~INIT3() 
{
}

void INIT3::setUp(VariantID vid)
{
}

void INIT3::runKernel(VariantID vid)
{
  Index_type run_size = getRunSize();
  const Index_type run_samples = getRunSamples();

  switch ( vid ) {

    case Baseline_Seq : {

      INIT3_DATA;

      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        for (Index_type i = 0; i < run_size; ++i ) {
          INIT3_BODY(i);
        }

      }
#endif
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      INIT3_DATA;

      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::simd_exec>(0, run_size, [=](int i) {
          INIT3_BODY(i);
        });

      }
#endif
      stopTimer();

      break;
    }

    case Baseline_OpenMP : {

      INIT3_DATA;

      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        #pragma omp for schedule(static)
        for (Index_type i = 0; i < run_size; ++i ) {
          INIT3_BODY(i);
        }

      }
#endif
      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {
      // case is not defined...
      break;
    }

    case RAJA_OpenMP : {

      INIT3_DATA;

      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(0, run_size, [=](int i) {
          INIT3_BODY(i);
        });

      }
#endif
      stopTimer();

      break;
    }

    case Baseline_CUDA : {

      INIT3_DATA;

      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

      }
#endif
      stopTimer();

      break;
    }

    case RAJA_CUDA : {

      INIT3_DATA;

      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

      }
#endif
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

void INIT3::updateChecksum(VariantID vid)
{
#if 0
  checksum[vid] += calcChecksum(m_p_new, getRunSize());
#endif
}

void INIT3::tearDown(VariantID vid)
{
  switch ( vid ) {

    case Baseline_Seq :
    case RAJA_Seq :
    case Baseline_OpenMP :
    case RAJA_OpenMP :
    case Baseline_CUDA :
    case RAJA_CUDA : {
      // De-allocate host memory here.
      break;
    }

#if 0
    case Baseline_OpenMP4x :
    case RAJA_OpenMP4x : {
      // De-allocate host and device memory here.
      break;
    }
#endif

    default : {
      std::cout << "\n  Unknown variant id = " << vid << std::endl;
    }

  }

  if (vid == Baseline_CUDA || vid == RAJA_CUDA) {
    // De-allocate device memory here.
  }
}

} // end namespace basic
} // end namespace rajaperf
