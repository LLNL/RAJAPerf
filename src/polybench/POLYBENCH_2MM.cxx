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
#if 0
  SizeSpec_T lsizespec = run_params.getSizeSpec();
   
  setDefaultSamples(100);
  switch(lsizespec) {
    case Mini:
      m_ni=16; m_nj=18; m_nk=22; m_nl=24;
      break;
    case Small:
      m_ni=40; m_nj=50; m_nk=70; m_nl=80;
      break;
    case Medium:
      m_ni=180; m_nj=190; m_nk=210; m_nl=220;
      break;
    case Large:
      m_ni=800; m_nj=900; m_nk=1100; m_nl=1200;
      break;
    case Extralarge:
      m_ni=1600; m_nj=1800; m_nk=2200; m_nl=2400;
      break;
    default:
      m_ni=180; m_nj=190; m_nk=210; m_nl=220;
      break;
  }
#endif  
}

POLYBENCH_2MM::~POLYBENCH_2MM() 
{
}

void POLYBENCH_2MM::setUp(VariantID vid)
{
  SizeSpec_T lsizespec = KernelBase::getSizeSpec();
   
  setDefaultSamples(100);
  switch(lsizespec) {
    case Mini:
      m_ni=16; m_nj=18; m_nk=22; m_nl=24;
      break;
    case Small:
      m_ni=40; m_nj=50; m_nk=70; m_nl=80;
      break;
    case Medium:
      m_ni=180; m_nj=190; m_nk=210; m_nl=220;
      break;
    case Large:
      m_ni=800; m_nj=900; m_nk=1100; m_nl=1200;
      break;
    case Extralarge:
      m_ni=1600; m_nj=1800; m_nk=2200; m_nl=2400;
      break;
    default:
      m_ni=180; m_nj=190; m_nk=210; m_nl=220;
      break;
  }
  allocAndInitData(m_tmp, m_ni * m_nj, vid);
  allocAndInitData(m_A, m_ni * m_nk, vid);
  allocAndInitData(m_B, m_nk * m_nj, vid);
  allocAndInitData(m_C, m_nj * m_nl, vid);
  allocAndInitData(m_D, m_ni * m_nl, vid);
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
  deallocData(m_tmp);
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_C);
  deallocData(m_D);
}

} // end namespace basic
} // end namespace rajaperf
