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


#include "ENERGY_CALC.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define ENERGY_CALC_DATA \
  ResReal_ptr e_new = m_e_new; \
  ResReal_ptr e_old = m_e_old; \
  ResReal_ptr delvc = m_delvc; \
  ResReal_ptr p_new = m_p_new; \
  ResReal_ptr p_old = m_p_old; \
  ResReal_ptr q_new = m_q_new; \
  ResReal_ptr q_old = m_q_old; \
  ResReal_ptr work = m_work; \
  ResReal_ptr compHalfStep = m_compHalfStep; \
  ResReal_ptr pHalfStep = m_pHalfStep; \
  ResReal_ptr bvc = m_bvc; \
  ResReal_ptr pbvc = m_pbvc; \
  ResReal_ptr ql_old = m_ql_old; \
  ResReal_ptr qq_old = m_qq_old; \
  ResReal_ptr vnewc = m_vnewc; \
  const Real_type rho0 = m_rho0; \
  const Real_type e_cut = m_e_cut; \
  const Real_type emin = m_emin; \
  const Real_type q_cut = m_q_cut;


#define ENERGY_CALC_BODY1 \
  e_new[i] = e_old[i] - 0.5 * delvc[i] * \
             (p_old[i] + q_old[i]) + 0.5 * work[i];

#define ENERGY_CALC_BODY2 \
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

#define ENERGY_CALC_BODY3 \
  e_new[i] = e_new[i] + 0.5 * delvc[i] \
             * ( 3.0*(p_old[i] + q_old[i]) \
                 - 4.0*(pHalfStep[i] + q_new[i])) ;

#define ENERGY_CALC_BODY4 \
  e_new[i] += 0.5 * work[i]; \
  if ( fabs(e_new[i]) < e_cut ) { e_new[i] = 0.0  ; } \
  if ( e_new[i]  < emin ) { e_new[i] = emin ; }

#define ENERGY_CALC_BODY5 \
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

#define ENERGY_CALC_BODY6 \
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
  setDefaultSamples(1300);
}

ENERGY_CALC::~ENERGY_CALC() 
{
}

void ENERGY_CALC::setUp(VariantID vid)
{
  allocAndInitData(m_e_new, getRunSize(), vid);
  allocAndInitData(m_e_old, getRunSize(), vid);
  allocAndInitData(m_delvc, getRunSize(), vid);
  allocAndInitData(m_p_new, getRunSize(), vid);
  allocAndInitData(m_p_old, getRunSize(), vid);
  allocAndInitData(m_q_new, getRunSize(), vid);
  allocAndInitData(m_q_old, getRunSize(), vid);
  allocAndInitData(m_work, getRunSize(), vid);
  allocAndInitData(m_compHalfStep, getRunSize(), vid);
  allocAndInitData(m_pHalfStep, getRunSize(), vid);
  allocAndInitData(m_bvc, getRunSize(), vid);
  allocAndInitData(m_pbvc, getRunSize(), vid);
  allocAndInitData(m_ql_old, getRunSize(), vid);
  allocAndInitData(m_qq_old, getRunSize(), vid);
  allocAndInitData(m_vnewc, getRunSize(), vid);
  
  initData(m_rho0);
  initData(m_e_cut);
  initData(m_emin);
  initData(m_q_cut);
}

void ENERGY_CALC::runKernel(VariantID vid)
{
  const Index_type run_samples = getRunSamples();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Baseline_Seq : {

      ENERGY_CALC_DATA;
  
      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_CALC_BODY1;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_CALC_BODY2;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_CALC_BODY3;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_CALC_BODY4;
        }
  
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_CALC_BODY5;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_CALC_BODY6;
        }

      }
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

      ENERGY_CALC_DATA;
 
      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          ENERGY_CALC_BODY1;
        }); 

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          ENERGY_CALC_BODY2;
        }); 

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          ENERGY_CALC_BODY3;
        }); 

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          ENERGY_CALC_BODY4;
        }); 

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          ENERGY_CALC_BODY5;
        }); 

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          ENERGY_CALC_BODY6;
        }); 

      }
      stopTimer(); 

      break;
    }

#if defined(_OPENMP)      
    case Baseline_OpenMP : {

      ENERGY_CALC_DATA;
 
      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        #pragma omp parallel
          {
            #pragma omp for nowait schedule(static)
            for (Index_type i = ibegin; i < iend; ++i ) {
              ENERGY_CALC_BODY1;
            }

            #pragma omp for nowait schedule(static)
            for (Index_type i = ibegin; i < iend; ++i ) {
              ENERGY_CALC_BODY2;
            }

            #pragma omp for nowait schedule(static)
            for (Index_type i = ibegin; i < iend; ++i ) {
              ENERGY_CALC_BODY3;
            }

            #pragma omp for nowait schedule(static)
            for (Index_type i = ibegin; i < iend; ++i ) {
              ENERGY_CALC_BODY4;
            }

            #pragma omp for nowait schedule(static)
            for (Index_type i = ibegin; i < iend; ++i ) {
              ENERGY_CALC_BODY5;
            }

            #pragma omp for nowait schedule(static)
            for (Index_type i = ibegin; i < iend; ++i ) {
              ENERGY_CALC_BODY6;
            }
          } // omp parallel

      }
      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {

      ENERGY_CALC_DATA;
      
      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
    
        #pragma omp parallel for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_CALC_BODY1;
        }

        #pragma omp parallel for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_CALC_BODY2;
        }

        #pragma omp parallel for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_CALC_BODY3;
        }

        #pragma omp parallel for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_CALC_BODY4;
        }

        #pragma omp parallel for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_CALC_BODY5;
        }

        #pragma omp parallel for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_CALC_BODY6;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      ENERGY_CALC_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          ENERGY_CALC_BODY1;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          ENERGY_CALC_BODY2;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          ENERGY_CALC_BODY3;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          ENERGY_CALC_BODY4;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          ENERGY_CALC_BODY5;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          ENERGY_CALC_BODY6;
        });

      }
      stopTimer();
      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Baseline_CUDA :
    case RAJA_CUDA : {
      // Fill these in later...you get the idea...
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

void ENERGY_CALC::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_e_new, getRunSize());
  checksum[vid] += calcChecksum(m_q_new, getRunSize());
}

void ENERGY_CALC::tearDown(VariantID vid)
{
  (void) vid;

  deallocData(m_e_new);
  deallocData(m_e_old);
  deallocData(m_delvc);
  deallocData(m_p_new);
  deallocData(m_p_old);
  deallocData(m_q_new);
  deallocData(m_q_old);
  deallocData(m_work);
  deallocData(m_compHalfStep);
  deallocData(m_pHalfStep);
  deallocData(m_bvc);
  deallocData(m_pbvc);
  deallocData(m_ql_old);
  deallocData(m_qq_old);
  deallocData(m_vnewc);
}

} // end namespace apps
} // end namespace rajaperf
