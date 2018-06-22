//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
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

#include "WIP-COUPLE.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"
#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define COUPLE_DATA_SETUP_CPU \
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

  switch ( vid ) {

    case Base_Seq : {

      COUPLE_DATA_SETUP_CPU;

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

      COUPLE_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(kmin, kmax), [=](int k) {
          COUPLE_BODY;
        }); 

      }
      stopTimer(); 

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)      
    case Base_OpenMP : {
      COUPLE_DATA_SETUP_CPU;

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

    case RAJA_OpenMP : {

      COUPLE_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(kmin, kmax), [=](int k) {
          COUPLE_BODY;
        }); 

      }
      stopTimer(); 

      break;
    }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP) && 0
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
      runOpenMPTargetVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA) && 0
    case Base_CUDA :
    case RAJA_CUDA :
    {
      runCudaVariant(vid);
      break;
    }
#endif

    default : {
      std::cout << "\n  COUPLE : Unknown variant id = " << vid << std::endl;
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
