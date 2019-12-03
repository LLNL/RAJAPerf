//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define PRESSURE_DATA_SETUP_CPU \
  ResReal_ptr compression = m_compression; \
  ResReal_ptr bvc = m_bvc; \
  ResReal_ptr p_new = m_p_new; \
  ResReal_ptr e_old  = m_e_old; \
  ResReal_ptr vnewc  = m_vnewc; \
  const Real_type cls = m_cls; \
  const Real_type p_cut = m_p_cut; \
  const Real_type pmin = m_pmin; \
  const Real_type eosvmax = m_eosvmax; 
   

PRESSURE::PRESSURE(const RunParams& params)
  : KernelBase(rajaperf::Apps_PRESSURE, params)
{
  setDefaultSize(100000);
  setDefaultReps(7000);
}

PRESSURE::~PRESSURE() 
{
}

void PRESSURE::setUp(VariantID vid)
{
  allocAndInitData(m_compression, getRunSize(), vid);
  allocAndInitData(m_bvc, getRunSize(), vid);
  allocAndInitDataConst(m_p_new, getRunSize(), 0.0, vid);
  allocAndInitData(m_e_old, getRunSize(), vid);
  allocAndInitData(m_vnewc, getRunSize(), vid);
  
  initData(m_cls);
  initData(m_p_cut);
  initData(m_pmin);
  initData(m_eosvmax);
}

void PRESSURE::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  PRESSURE_DATA_SETUP_CPU;

  auto pressure_lam1 = [=](int i) {
                         PRESSURE_BODY1;
                       };
  auto pressure_lam2 = [=](int i) {
                         PRESSURE_BODY2;
                       };
  
  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          PRESSURE_BODY1;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          PRESSURE_BODY2;
        }

      }
      stopTimer();

      break;
    } 

#if defined(RUN_RAJA_SEQ)     
    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::seq_region>( [=]() {

          RAJA::forall<RAJA::loop_exec>(
            RAJA::RangeSegment(ibegin, iend), pressure_lam1);

          RAJA::forall<RAJA::loop_exec>(
            RAJA::RangeSegment(ibegin, iend), pressure_lam2);

        }); // end sequential region (for single-source code)

      }
      stopTimer(); 

      break;
    }
#endif // RUN_RAJA_SEQ

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
        {

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            PRESSURE_BODY1;
          }

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            PRESSURE_BODY2;
          }

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case OpenMP_Lambda : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
        {

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            pressure_lam1(i);
          }

          #pragma omp for nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            pressure_lam2(i);
          }

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), pressure_lam1);

          RAJA::forall<RAJA::omp_for_nowait_exec>(
            RAJA::RangeSegment(ibegin, iend), pressure_lam2);

        }); // end omp parallel region

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
      std::cout << "\n  PRESSURE : Unknown variant id = " << vid << std::endl;
    }

  }
}

void PRESSURE::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_p_new, getRunSize());
}

void PRESSURE::tearDown(VariantID vid)
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
