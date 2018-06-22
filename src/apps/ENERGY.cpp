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

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define ENERGY_DATA_SETUP_CPU \
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


ENERGY::ENERGY(const RunParams& params)
  : KernelBase(rajaperf::Apps_ENERGY, params)
{
  setDefaultSize(100000);
  setDefaultReps(1300);
}

ENERGY::~ENERGY() 
{
}

void ENERGY::setUp(VariantID vid)
{
  allocAndInitDataConst(m_e_new, getRunSize(), 0.0, vid);
  allocAndInitData(m_e_old, getRunSize(), vid);
  allocAndInitData(m_delvc, getRunSize(), vid);
  allocAndInitData(m_p_new, getRunSize(), vid);
  allocAndInitData(m_p_old, getRunSize(), vid);
  allocAndInitDataConst(m_q_new, getRunSize(), 0.0, vid);
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

void ENERGY::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      ENERGY_DATA_SETUP_CPU;
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY1;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY2;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY3;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY4;
        }
  
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY5;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY6;
        }

      }
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

      ENERGY_DATA_SETUP_CPU;
 
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](int i) {
          ENERGY_BODY1;
        }); 

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](int i) {
          ENERGY_BODY2;
        }); 

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](int i) {
          ENERGY_BODY3;
        }); 

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](int i) {
          ENERGY_BODY4;
        }); 

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](int i) {
          ENERGY_BODY5;
        }); 

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](int i) {
          ENERGY_BODY6;
        }); 

      }
      stopTimer(); 

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)      
    case Base_OpenMP : {

//
// NOTE: This kernel should be written to have an OpenMP parallel 
//       region around it and then use an OpenMP for-nowait for
//       each loop inside it. We currently don't have a clean way to
//       do this in RAJA. So, the base OpenMP variant is coded the
//       way it is to be able to do an "apples to apples" comparison.
//
//       This will be changed in the future when the required feature 
//       is added to RAJA.
//

      ENERGY_DATA_SETUP_CPU;
      
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    
        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY1;
        }

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY2;
        }

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY3;
        }

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY4;
        }

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY5;
        }

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY6;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      ENERGY_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](int i) {
          ENERGY_BODY1;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](int i) {
          ENERGY_BODY2;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](int i) {
          ENERGY_BODY3;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](int i) {
          ENERGY_BODY4;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](int i) {
          ENERGY_BODY5;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](int i) {
          ENERGY_BODY6;
        });

      }
      stopTimer();
      break;
    }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
      runOpenMPTargetVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA :
    case RAJA_CUDA :
    {
      runCudaVariant(vid);
      break;
    }
#endif

    default : {
      std::cout << "\n  ENERGY : Unknown variant id = " << vid << std::endl;
    }

  }
}

void ENERGY::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_e_new, getRunSize());
  checksum[vid] += calcChecksum(m_q_new, getRunSize());
}

void ENERGY::tearDown(VariantID vid)
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
