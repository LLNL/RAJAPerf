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

#include "RAJA/RAJA.hxx"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define PRESSURE_CALC_DATA \
  ResReal_ptr compression = m_compression; \
  ResReal_ptr bvc = m_bvc; \
  ResReal_ptr p_new = m_p_new; \
  ResReal_ptr e_old  = m_e_old; \
  ResReal_ptr vnewc  = m_vnewc; \
  const Real_type cls = m_cls; \
  const Real_type p_cut = m_p_cut; \
  const Real_type pmin = m_pmin; \
  const Real_type eosvmax = m_eosvmax; 
   

#define PRESSURE_CALC_BODY1 \
  bvc[i] = cls * (compression[i] + 1.0);

#define PRESSURE_CALC_BODY2 \
  p_new[i] = bvc[i] * e_old[i] ; \
  if ( fabs(p_new[i]) <  p_cut ) p_new[i] = 0.0 ; \
  if ( vnewc[i] >= eosvmax ) p_new[i] = 0.0 ; \
  if ( p_new[i]  <  pmin ) p_new[i]   = pmin ;


PRESSURE_CALC::PRESSURE_CALC(const RunParams& params)
  : KernelBase(rajaperf::Apps_PRESSURE_CALC, params)
{
  setDefaultSize(100000);
  setDefaultSamples(7000);
}

PRESSURE_CALC::~PRESSURE_CALC() 
{
}

void PRESSURE_CALC::setUp(VariantID vid)
{
  allocAndInitData(m_compression, getRunSize(), vid);
  allocAndInitData(m_bvc, getRunSize(), vid);
  allocAndInitData(m_p_new, getRunSize(), vid);
  allocAndInitData(m_e_old, getRunSize(), vid);
  allocAndInitData(m_vnewc, getRunSize(), vid);
  
  initData(m_cls);
  initData(m_p_cut);
  initData(m_pmin);
  initData(m_eosvmax);
}

void PRESSURE_CALC::runKernel(VariantID vid)
{
  const Index_type run_samples = getRunSamples();
  const Index_type lbegin = 0;
  const Index_type lend = getRunSize();

  switch ( vid ) {

    case Baseline_Seq : {

      PRESSURE_CALC_DATA;
  
      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        for (Index_type i = lbegin; i < lend; ++i ) {
          PRESSURE_CALC_BODY1;
        }

        for (Index_type i = lbegin; i < lend; ++i ) {
          PRESSURE_CALC_BODY2;
        }

      }
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

      PRESSURE_CALC_DATA;
 
      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::simd_exec>(lbegin, lend, [=](int i) {
          PRESSURE_CALC_BODY1;
        }); 

        RAJA::forall<RAJA::simd_exec>(lbegin, lend, [=](int i) {
          PRESSURE_CALC_BODY2;
        }); 

      }
      stopTimer(); 

      break;
    }

    case Baseline_OpenMP : {
#if defined(_OPENMP)      
      PRESSURE_CALC_DATA;
 
      startTimer();
      for (SampIndex_type isamp = lbegin; isamp < run_samples; ++isamp) {

        #pragma omp parallel
          {
            #pragma omp for nowait schedule(static)
            for (Index_type i = lbegin; i < lend; ++i ) {
              PRESSURE_CALC_BODY1;
            }

            #pragma omp for nowait schedule(static)
            for (Index_type i = lbegin; i < lend; ++i ) {
              PRESSURE_CALC_BODY2;
            }
          } // omp parallel

      }
      stopTimer();
#endif
      break;
    }

    case RAJALike_OpenMP : {
#if defined(_OPENMP)      
      PRESSURE_CALC_DATA;
      
      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
    
        #pragma omp parallel for schedule(static)
        for (Index_type i = lbegin; i < lend; ++i ) {
          PRESSURE_CALC_BODY1;
        }

        #pragma omp parallel for schedule(static)
        for (Index_type i = lbegin; i < lend; ++i ) {
          PRESSURE_CALC_BODY2;
        }

      }
      stopTimer();
#endif
      break;
    }

    case RAJA_OpenMP : {
#if defined(_OPENMP)      

      PRESSURE_CALC_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(lbegin, lend, [=](int i) {
          PRESSURE_CALC_BODY1;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(lbegin, lend, [=](int i) {
          PRESSURE_CALC_BODY2;
        });

      }
      stopTimer();
#endif
      break;
    }

    case Baseline_CUDA :
    case RAJA_CUDA : {
      // Fill these in later...you get the idea...
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
  (void) vid;
 
  deallocData(m_compression);
  deallocData(m_bvc);
  deallocData(m_p_new);
  deallocData(m_e_old);
  deallocData(m_vnewc);
}

} // end namespace apps
} // end namespace rajaperf
