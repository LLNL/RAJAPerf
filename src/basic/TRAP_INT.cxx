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

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define TRAP_INT_DATA 

#define TRAP_INT_BODY


TRAP_INT::TRAP_INT(const RunParams& params)
  : KernelBase(rajaperf::Basic_TRAP_INT, params)
{
   setDefaultSize(100000);
   setDefaultSamples(10000);
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
#if 0
  Index_type run_size = getRunSize();
#endif
  const Index_type run_samples = getRunSamples();

  switch ( vid ) {

    case Baseline_Seq : {

      TRAP_INT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
        for (Index_type i = 0; i < run_size; ++i ) {
          TRAP_INT_BODY;
        }
#endif

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      TRAP_INT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
        RAJA::forall<RAJA::simd_exec>(0, run_size, [=](int i) {
          TRAP_INT_BODY;
        });
#endif

      }
      stopTimer();

      break;
    }

    case Baseline_OpenMP : {

      TRAP_INT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
        #pragma omp for schedule(static)
        for (Index_type i = 0; i < run_size; ++i ) {
          TRAP_INT_BODY;
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

      TRAP_INT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
        RAJA::forall<RAJA::omp_parallel_for_exec>(0, run_size, [=](int i) {
          TRAP_INT_BODY;
        });
#endif

      }
      stopTimer();

      break;
    }

    case Baseline_CUDA : {

      TRAP_INT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
#endif
      }
      stopTimer();

      break;
    }

    case RAJA_CUDA : {

      TRAP_INT_DATA;

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

void TRAP_INT::updateChecksum(VariantID vid)
{
  (void) vid;
#if 0
  checksum[vid] += calcChecksum(m_p_new, getRunSize());
#endif
}

void TRAP_INT::tearDown(VariantID vid)
{
  (void) vid;
}

} // end namespace basic
} // end namespace rajaperf
