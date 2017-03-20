/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for kernel PRESSURE_CALC.
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


#include "PRESSURE_CALC.hxx"

#include "common/DataUtils.hxx"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define PRESSURE_CALC_DATA \
  RAJA::Real_ptr compression = m_compression; \
  RAJA::Real_ptr bvc = m_bvc; \
  RAJA::Real_ptr p_new = m_p_new; \
  RAJA::Real_ptr e_old  = m_e_old; \
  RAJA::Real_ptr vnewc  = m_vnewc; \
  const RAJA::Real_type cls = m_cls; \
  const RAJA::Real_type p_cut = m_p_cut; \
  const RAJA::Real_type pmin = m_pmin; \
  const RAJA::Real_type eosvmax = m_eosvmax;
   

#define PRESSURE_CALC_BODY1(i) \
  bvc[i] = cls * (compression[i] + 1.0);

#define PRESSURE_CALC_BODY2(i) \
  p_new[i] = bvc[i] * e_old[i] ; \
  if ( fabs(p_new[i]) <  p_cut ) p_new[i] = 0.0 ; \
  if ( vnewc[i] >= eosvmax ) p_new[i] = 0.0 ; \
  if ( p_new[i]  <  pmin ) p_new[i]   = pmin ;


PRESSURE_CALC::PRESSURE_CALC(const RunParams& params)
  : KernelBase(rajaperf::Apps_PRESSURE_CALC, params)
{
  setDefaultSize(100000);
  setDefaultSamples(10000);
}

PRESSURE_CALC::~PRESSURE_CALC() 
{
}

void PRESSURE_CALC::setUp(VariantID vid)
{
  allocAndInitAligned(m_compression, getRunSize(), vid);
  allocAndInitAligned(m_bvc, getRunSize(), vid);
  allocAndInitAligned(m_p_new, getRunSize(), vid);
  allocAndInitAligned(m_e_old, getRunSize(), vid);
  allocAndInitAligned(m_vnewc, getRunSize(), vid);
  
  initData(m_cls);
  initData(m_p_cut);
  initData(m_pmin);
  initData(m_eosvmax);
}

void PRESSURE_CALC::runKernel(VariantID vid)
{
  int run_size = getRunSize();
  int run_samples = getRunSamples();

  switch ( vid ) {

    case Baseline_Seq : {

      PRESSURE_CALC_DATA;
  
      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          PRESSURE_CALC_BODY1(i);
        }

        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          PRESSURE_CALC_BODY2(i);
        }

      }
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

      PRESSURE_CALC_DATA;
 
      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::simd_exec>(0, run_size, [=](int i) {
          PRESSURE_CALC_BODY1(i);
        }); 

        RAJA::forall<RAJA::simd_exec>(0, run_size, [=](int i) {
          PRESSURE_CALC_BODY2(i);
        }); 

      }
#endif
      stopTimer(); 

      break;
    }

    case Baseline_OpenMP : {
      
      PRESSURE_CALC_DATA;
 
      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        #pragma omp parallel
          {
            #pragma omp for nowait schedule(static)
            for (RAJA::Index_type i = 0; i < run_size; ++i ) {
              PRESSURE_CALC_BODY1(i);
            }

            #pragma omp for nowait schedule(static)
            for (RAJA::Index_type i = 0; i < run_size; ++i ) {
              PRESSURE_CALC_BODY2(i);
            }
          } // omp parallel

      }
#endif
      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {

      PRESSURE_CALC_DATA;
      
      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
    
        #pragma omp parallel for schedule(static)
        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          PRESSURE_CALC_BODY1(i);
        }

        #pragma omp parallel for schedule(static)
        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          PRESSURE_CALC_BODY2(i);
        }

      }
#endif
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      PRESSURE_CALC_DATA;

      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(0, run_size, [=](int i) {
          PRESSURE_CALC_BODY1(i);
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(0, run_size, [=](int i) {
          PRESSURE_CALC_BODY2(i);
        });

      }
#endif
      stopTimer();

      break;
    }

    case Baseline_CUDA :
    case RAJA_CUDA : {
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

void PRESSURE_CALC::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_p_new, getRunSize());
}

void PRESSURE_CALC::tearDown(VariantID vid)
{
  freeAligned(m_compression);
  freeAligned(m_bvc);
  freeAligned(m_p_new);
  freeAligned(m_e_old);
  freeAligned(m_vnewc);
  
  if (vid == Baseline_CUDA || vid == RAJA_CUDA) {
    // De-allocate device memory here.
  }
}

} // end namespace apps
} // end namespace rajaperf
