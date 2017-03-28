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


#include "IF_QUAD.hxx"

#include "common/DataUtils.hxx"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define IF_QUAD_DATA 

#define IF_QUAD_BODY


IF_QUAD::IF_QUAD(const RunParams& params)
  : KernelBase(rajaperf::Basic_IF_QUAD, params)
{
   setDefaultSize(100000);
   setDefaultSamples(10000);
}

IF_QUAD::~IF_QUAD() 
{
}

void IF_QUAD::setUp(VariantID vid)
{
  (void) vid;
}

void IF_QUAD::runKernel(VariantID vid)
{
#if 0
  Index_type run_size = getRunSize();
#endif
  const Index_type run_samples = getRunSamples();

  switch ( vid ) {

    case Baseline_Seq : {

      IF_QUAD_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
        for (Index_type i = 0; i < run_size; ++i ) {
          IF_QUAD_BODY;
        }
#endif

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      IF_QUAD_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
        RAJA::forall<RAJA::simd_exec>(0, run_size, [=](int i) {
          IF_QUAD_BODY;
        });
#endif

      }
      stopTimer();

      break;
    }

    case Baseline_OpenMP : {

      IF_QUAD_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
        #pragma omp for schedule(static)
        for (Index_type i = 0; i < run_size; ++i ) {
          IF_QUAD_BODY;
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

      IF_QUAD_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
        RAJA::forall<RAJA::omp_parallel_for_exec>(0, run_size, [=](int i) {
          IF_QUAD_BODY;
        });
#endif

      }
      stopTimer();

      break;
    }

    case Baseline_CUDA : {

      IF_QUAD_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
#if 0
#endif
      }
      stopTimer();

      break;
    }

    case RAJA_CUDA : {

      IF_QUAD_DATA;

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

void IF_QUAD::updateChecksum(VariantID vid)
{
  (void) vid;
#if 0
  checksum[vid] += calcChecksum(m_p_new, getRunSize());
#endif
}

void IF_QUAD::tearDown(VariantID vid)
{
  (void) vid;
}

} // end namespace basic
} // end namespace rajaperf
