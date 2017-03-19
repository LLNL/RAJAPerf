/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for kernel ENERGY_CALC.
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


#include "ENERGY_CALC.hxx"

#include "common/DataUtils.hxx"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define ENERGY_CALC_DATA \
  RAJA::Real_ptr e_new = m_e_new; \
  RAJA::Real_ptr e_old = m_e_old; \
  RAJA::Real_ptr delvc = m_delvc; \
  RAJA::Real_ptr p_new = m_p_new; \
  RAJA::Real_ptr p_old = m_p_old; \
  RAJA::Real_ptr q_new = m_q_new; \
  RAJA::Real_ptr q_old = m_q_old; \
  RAJA::Real_ptr work = m_work; \
  RAJA::Real_ptr compHalfStep = m_compHalfStep; \
  RAJA::Real_ptr pHalfStep = m_pHalfStep; \
  RAJA::Real_ptr bvc = m_bvc; \
  RAJA::Real_ptr pbvc = m_pbvc; \
  RAJA::Real_ptr ql_old = m_ql_old; \
  RAJA::Real_ptr qq_old = m_qq_old; \
  RAJA::Real_ptr vnewc = m_vnewc; \
  RAJA::Real_type rho0 = m_rho0; \
  RAJA::Real_type e_cut = m_e_cut; \
  RAJA::Real_type emin = m_emin; \
  RAJA::Real_type q_cut = m_q_cut;
   

#define ENERGY_CALC_BODY1(i) \
  e_new[i] = e_old[i] - 0.5 * delvc[i] * \
             (p_old[i] + q_old[i]) + 0.5 * work[i];

#define ENERGY_CALC_BODY2(i) \
  if ( delvc[i] > 0.0 ) { \
     q_new[i] = 0.0 ; \
  } \
  else { \
     Real_type vhalf = 1.0 / (1.0 + compHalfStep[i]) ; \
     Real_type ssc = ( pbvc[i] * e_new[i] \
        + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_new[i] = (ssc*ql_old[i] + qq_old[i]) ; \
  }

#define ENERGY_CALC_BODY3(i) \
  e_new[i] = e_new[i] + 0.5 * delvc[i] \
             * ( 3.0*(p_old[i] + q_old[i]) \
                 - 4.0*(pHalfStep[i] + q_new[i])) ;

#define ENERGY_CALC_BODY4(i) \
  e_new[i] += 0.5 * work[i]; \
  if ( fabs(e_new[i]) < e_cut ) { e_new[i] = 0.0  ; } \
  if ( e_new[i]  < emin ) { e_new[i] = emin ; }

#define ENERGY_CALC_BODY5(i) \
  Real_type q_tilde ; \
  if (delvc[i] > 0.0) { \
     q_tilde = 0. ; \
  } \
  else { \
     Real_type ssc = ( pbvc[i] * e_new[i] \
         + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_tilde = (ssc*ql_old[i] + qq_old[i]) ; \
  } \
  e_new[i] = e_new[i] - ( 7.0*(p_old[i] + q_old[i]) \
                         - 8.0*(pHalfStep[i] + q_new[i]) \
                         + (p_new[i] + q_tilde)) * delvc[i] / 6.0 ; \
  if ( fabs(e_new[i]) < e_cut ) { \
     e_new[i] = 0.0  ; \
  } \
  if ( e_new[i]  < emin ) { \
     e_new[i] = emin ; \
  }

#define ENERGY_CALC_BODY6(i) \
  if ( delvc[i] <= 0.0 ) { \
     Real_type ssc = ( pbvc[i] * e_new[i] \
             + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_new[i] = (ssc*ql_old[i] + qq_old[i]) ; \
     if (fabs(q_new[i]) < q_cut) q_new[i] = 0.0 ; \
  }


ENERGY_CALC::ENERGY_CALC(const RunParams& params)
  : KernelBase(rajaperf::Apps_ENERGY_CALC, params)
{
  setDefaultSize(100000);
  setDefaultSamples(2000);
}

ENERGY_CALC::~ENERGY_CALC() 
{
}

//
// NOTE: Setup and execute methods are implemented using switch statements
//       for now. We may want to break the variants out differently...
//
void ENERGY_CALC::setUp(VariantID vid)
{
  allocAndInitAligned(m_e_new, getRunSize());
  allocAndInitAligned(m_e_old, getRunSize());
  allocAndInitAligned(m_delvc, getRunSize());
  allocAndInitAligned(m_p_new, getRunSize());
  allocAndInitAligned(m_p_old, getRunSize());
  allocAndInitAligned(m_q_new, getRunSize());
  allocAndInitAligned(m_q_old, getRunSize());
  allocAndInitAligned(m_work, getRunSize());
  allocAndInitAligned(m_compHalfStep, getRunSize());
  allocAndInitAligned(m_pHalfStep, getRunSize());
  allocAndInitAligned(m_bvc, getRunSize());
  allocAndInitAligned(m_pbvc, getRunSize());
  allocAndInitAligned(m_ql_old, getRunSize());
  allocAndInitAligned(m_qq_old, getRunSize());
  allocAndInitAligned(m_vnewc, getRunSize());
  
  initData(m_rho0);
  initData(m_e_cut);
  initData(m_emin);
  initData(m_q_cut);
}

void ENERGY_CALC::runKernel(VariantID vid)
{
  int run_size = getRunSize();
  int run_samples = getRunSamples();

  switch ( vid ) {

    case Baseline_Seq : {

      ENERGY_CALC_DATA;
  
      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          ENERGY_CALC_BODY1(i);
        }

        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          ENERGY_CALC_BODY2(i);
        }

        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          ENERGY_CALC_BODY3(i);
        }

        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          ENERGY_CALC_BODY4(i);
        }
  
        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          ENERGY_CALC_BODY5(i);
        }

        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          ENERGY_CALC_BODY6(i);
        }

      }
#endif
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

      ENERGY_CALC_DATA;
 
      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::simd_exec>(0, run_size, [=](int i) {
          ENERGY_CALC_BODY1(i);
        }); 

        RAJA::forall<RAJA::simd_exec>(0, run_size, [=](int i) {
          ENERGY_CALC_BODY2(i);
        }); 

        RAJA::forall<RAJA::simd_exec>(0, run_size, [=](int i) {
          ENERGY_CALC_BODY3(i);
        }); 

        RAJA::forall<RAJA::simd_exec>(0, run_size, [=](int i) {
          ENERGY_CALC_BODY4(i);
        }); 

        RAJA::forall<RAJA::simd_exec>(0, run_size, [=](int i) {
          ENERGY_CALC_BODY5(i);
        }); 

        RAJA::forall<RAJA::simd_exec>(0, run_size, [=](int i) {
          ENERGY_CALC_BODY6(i);
        }); 

      }
#endif
      stopTimer(); 

      break;
    }

    case Baseline_OpenMP : {
      
      ENERGY_CALC_DATA;
 
      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        #pragma omp parallel
          {
            #pragma omp for nowait schedule(static)
            for (RAJA::Index_type i = 0; i < run_size; ++i ) {
              ENERGY_CALC_BODY1(i);
            }

            #pragma omp for nowait schedule(static)
            for (RAJA::Index_type i = 0; i < run_size; ++i ) {
              ENERGY_CALC_BODY2(i);
            }

            #pragma omp for nowait schedule(static)
            for (RAJA::Index_type i = 0; i < run_size; ++i ) {
              ENERGY_CALC_BODY3(i);
            }

            #pragma omp for nowait schedule(static)
            for (RAJA::Index_type i = 0; i < run_size; ++i ) {
              ENERGY_CALC_BODY4(i);
            }

            #pragma omp for nowait schedule(static)
            for (RAJA::Index_type i = 0; i < run_size; ++i ) {
              ENERGY_CALC_BODY5(i);
            }

            #pragma omp for nowait schedule(static)
            for (RAJA::Index_type i = 0; i < run_size; ++i ) {
              ENERGY_CALC_BODY6(i);
            }
          } // omp parallel

      }
#endif
      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {

      ENERGY_CALC_DATA;
      
      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
    
        #pragma omp parallel for schedule(static)
        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          ENERGY_CALC_BODY1(i);
        }

        #pragma omp parallel for schedule(static)
        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          ENERGY_CALC_BODY2(i);
        }

        #pragma omp parallel for schedule(static)
        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          ENERGY_CALC_BODY3(i);
        }

        #pragma omp parallel for schedule(static)
        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          ENERGY_CALC_BODY4(i);
        }

        #pragma omp parallel for schedule(static)
        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          ENERGY_CALC_BODY5(i);
        }

        #pragma omp parallel for schedule(static)
        for (RAJA::Index_type i = 0; i < run_size; ++i ) {
          ENERGY_CALC_BODY6(i);
        }

      }
#endif
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      ENERGY_CALC_DATA;

      startTimer();
#if 0
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(0, run_size, [=](int i) {
          ENERGY_CALC_BODY1(i);
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(0, run_size, [=](int i) {
          ENERGY_CALC_BODY2(i);
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(0, run_size, [=](int i) {
          ENERGY_CALC_BODY3(i);
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(0, run_size, [=](int i) {
          ENERGY_CALC_BODY4(i);
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(0, run_size, [=](int i) {
          ENERGY_CALC_BODY5(i);
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(0, run_size, [=](int i) {
          ENERGY_CALC_BODY6(i);
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

void ENERGY_CALC::updateChecksum(VariantID vid)
{
#if 0
  checksum[vid] += calcChecksum(m_e_new, getRunSize());
  checksum[vid] += calcChecksum(m_q_new, getRunSize());
#endif
}

void ENERGY_CALC::tearDown(VariantID vid)
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

} // end namespace apps
} // end namespace rajaperf
