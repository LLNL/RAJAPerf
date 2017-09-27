//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// COUPLE kernel reference implementation:
///
/// for (Index_type k = kmin ; k < kmax ; ++k ) {
///   for (Index_type j = jmin; j < jmax; j++) { 
///    
///      Index_type it0=    ((k)*(jmax+1) + (j))*(imax+1) ; 
///      Index_type idenac= ((k)*(jmax+2) + (j))*(imax+2) ; 
///    
///      for (Index_type i = imin; i < imax; i++) { 
///    
///         Complex_type c1 = c10 * denac[idenac+i]; 
///         Complex_type c2 = c20 * denlw[it0+i]; 
///    
///         /* promote to doubles to avoid possible divide by zero */ 
///         Real_type c1re = real(c1);  Real_type c1im = imag(c1); 
///         Real_type c2re = real(c2);  Real_type c2im = imag(c2); 
///    
///         /* lamda = sqrt(|c1|^2 + |c2|^2) uses doubles to avoid underflow. */
///         Real_type zlam = c1re*c1re + c1im*c1im + 
///                          c2re*c2re + c2im*c2im + 1.0e-34; 
///         zlam = sqrt(zlam); 
///         Real_type snlamt = sin(zlam * dt * 0.5); 
///         Real_type cslamt = cos(zlam * dt * 0.5); 
///    
///         Complex_type a0t = t0[it0+i]; 
///         Complex_type a1t = t1[it0+i]; 
///         Complex_type a2t = t2[it0+i] * fratio; 
///    
///         Real_type r_zlam= 1.0/zlam; 
///         c1 *= r_zlam; 
///         c2 *= r_zlam; 
///         Real_type zac1 = zabs2(c1); 
///         Real_type zac2 = zabs2(c2); 
///    
///         /* compute new A0 */ 
///         Complex_type z3 = ( c1 * a1t + c2 * a2t ) * snlamt ; 
///         t0[it0+i] = a0t * cslamt -  ireal * z3; 
///    
///         /* compute new A1  */ 
///         Real_type r = zac1 * cslamt + zac2; 
///         Complex_type z5 = c2 * a2t; 
///         Complex_type z4 = conj(c1) * z5 * (cslamt-1); 
///         z3 = conj(c1) * a0t * snlamt; 
///         t1[it0+i] = a1t * r + z4 - ireal * z3; 
///    
///         /* compute new A2  */ 
///         r = zac1 + zac2 * cslamt; 
///         z5 = c1 * a1t; 
///         z4 = conj(c2) * z5 * (cslamt-1); 
///         z3 = conj(c2) * a0t * snlamt; 
///         t2[it0+i] = ( a2t * r + z4 - ireal * z3 ) * r_fratio; 
///    
///      } /* i loop */ 
///    
///   } /* j loop */
/// } /* k loop */
///

#include "WIP-COUPLE.hpp"

#include "AppsData.hpp"
#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define COUPLE_DATA \
  ResComplex_ptr t0 = m_t0; \
  ResComplex_ptr t1 = m_t1; \
  ResComplex_ptr t2 = m_t2; \
  ResComplex_ptr denac = m_denac; \
  ResComplex_ptr denlw = m_denlw; \
  const Real_type dt = m_dt; \
  const Real_type c10 = m_c10; \
  const Real_type fratio = m_fratio; \
  const Real_type r_fratio = m_r_fratio; \
  const Real_type c20 = m_c20; \
  const Complex_type ireal = m_ireal; \
 \
  const Index_type imin = m_imin; \
  const Index_type imax = m_imax; \
  const Index_type jmin = m_jmin; \
  const Index_type jmax = m_jmax; \
  const Index_type kmin = m_kmin; \
  const Index_type kmax = m_kmax;


#define COUPLE_BODY \
for (Index_type j = jmin; j < jmax; j++) { \
 \
   Index_type it0=    ((k)*(jmax+1) + (j))*(imax+1) ; \
   Index_type idenac= ((k)*(jmax+2) + (j))*(imax+2) ; \
 \
   for (Index_type i = imin; i < imax; i++) { \
 \
      Complex_type c1 = c10 * denac[idenac+i]; \
      Complex_type c2 = c20 * denlw[it0+i]; \
 \
      /* promote to doubles to avoid possible divide by zero */ \
      Real_type c1re = real(c1);  Real_type c1im = imag(c1); \
      Real_type c2re = real(c2);  Real_type c2im = imag(c2); \
 \
      /* lamda = sqrt(|c1|^2 + |c2|^2) uses doubles to avoid underflow. */ \
      Real_type zlam = c1re*c1re + c1im*c1im + \
                       c2re*c2re + c2im*c2im + 1.0e-34; \
      zlam = sqrt(zlam); \
      Real_type snlamt = sin(zlam * dt * 0.5); \
      Real_type cslamt = cos(zlam * dt * 0.5); \
 \
      Complex_type a0t = t0[it0+i]; \
      Complex_type a1t = t1[it0+i]; \
      Complex_type a2t = t2[it0+i] * fratio; \
 \
      Real_type r_zlam= 1.0/zlam; \
      c1 *= r_zlam; \
      c2 *= r_zlam; \
      Real_type zac1 = zabs2(c1); \
      Real_type zac2 = zabs2(c2); \
 \
      /* compute new A0 */ \
      Complex_type z3 = ( c1 * a1t + c2 * a2t ) * snlamt ; \
      t0[it0+i] = a0t * cslamt -  ireal * z3; \
 \
      /* compute new A1  */ \
      Real_type r = zac1 * cslamt + zac2; \
      Complex_type z5 = c2 * a2t; \
      Complex_type z4 = conj(c1) * z5 * (cslamt-1); \
      z3 = conj(c1) * a0t * snlamt; \
      t1[it0+i] = a1t * r + z4 - ireal * z3; \
 \
      /* compute new A2  */ \
      r = zac1 + zac2 * cslamt; \
      z5 = c1 * a1t; \
      z4 = conj(c2) * z5 * (cslamt-1); \
      z3 = conj(c2) * a0t * snlamt; \
      t2[it0+i] = ( a2t * r + z4 - ireal * z3 ) * r_fratio; \
 \
   } /* i loop */ \
 \
} /* j loop */


COUPLE::COUPLE(const RunParams& params)
  : KernelBase(rajaperf::Apps_COUPLE, params)
{
  setDefaultSize(64);  // See rzmax in ADomain struct
  setDefaultReps(60);

  m_domain = new ADomain(getRunSize(), /* ndims = */ 3);

  m_imin = m_domain->imin;
  m_imax = m_domain->imax;
  m_jmin = m_domain->jmin;
  m_jmax = m_domain->jmax;
  m_kmin = m_domain->kmin;
  m_kmax = m_domain->kmax;
}

COUPLE::~COUPLE() 
{
  delete m_domain;
}

Index_type COUPLE::getItsPerRep() const 
{ 
  return  ( (m_imax - m_imin) * (m_jmax - m_jmin) * (m_kmax - m_kmin) ); 
}

void COUPLE::setUp(VariantID vid)
{
  int max_loop_index = m_domain->lrn;

  allocAndInitData(m_t0, max_loop_index, vid);
  allocAndInitData(m_t1, max_loop_index, vid);
  allocAndInitData(m_t2, max_loop_index, vid);
  allocAndInitData(m_denac, max_loop_index, vid);
  allocAndInitData(m_denlw, max_loop_index, vid);

  m_clight = 3.e+10;
  m_csound = 3.09e+7;
  m_omega0 = 0.9;
  m_omegar = 0.9;
  m_dt = 0.208;
  m_c10 = 0.25 * (m_clight / m_csound);
  m_fratio = sqrt(m_omegar / m_omega0);
  m_r_fratio = 1.0/m_fratio;
  m_c20 = 0.25 * (m_clight / m_csound) * m_r_fratio;
  m_ireal = Complex_type(0.0, 1.0); 
}

void COUPLE::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();

//
// RDH: Should we use forallN for this kernel???
//

  switch ( vid ) {

    case Base_Seq : {

      COUPLE_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = kmin ; k < kmax ; ++k ) {
          COUPLE_BODY;
        }

      }
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

      COUPLE_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::seq_exec>(kmin, kmax, [=](int k) {
          COUPLE_BODY;
        }); 

      }
      stopTimer(); 

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)      
    case Base_OpenMP : {
      COUPLE_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for 
        for (Index_type k = kmin ; k < kmax ; ++k ) {
          COUPLE_BODY;
        }

      }
      stopTimer();
      break;
    }

    case RAJALike_OpenMP : {
      // Not applicable
      break;
    }

    case RAJA_OpenMP : {

      COUPLE_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(kmin, kmax, [=](int k) {
          COUPLE_BODY;
        }); 

      }
      stopTimer(); 

      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA :
    case RAJA_CUDA : {
      // Fill these in later...you get the idea...
      break;
    }
#endif

#if 0
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget : {
      // Fill these in later...you get the idea...
      break;
    }
#endif

    default : {
      std::cout << "\n  Unknown variant id = " << vid << std::endl;
    }

  }
}

void COUPLE::updateChecksum(VariantID vid)
{
  int max_loop_index = m_domain->lrn;

  checksum[vid] += calcChecksum(m_t0, max_loop_index);
  checksum[vid] += calcChecksum(m_t1, max_loop_index);
  checksum[vid] += calcChecksum(m_t2, max_loop_index);
}

void COUPLE::tearDown(VariantID vid)
{
  (void) vid;
 
  deallocData(m_t0);
  deallocData(m_t1);
  deallocData(m_t2);
  deallocData(m_denac);
  deallocData(m_denlw);
}

} // end namespace apps
} // end namespace rajaperf
