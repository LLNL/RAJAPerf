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

#include "PLANCKIAN.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>
#include <cmath>

namespace rajaperf 
{
namespace lcals
{


#define PLANCKIAN_DATA_SETUP_CPU \
  ResReal_ptr x = m_x; \
  ResReal_ptr y = m_y; \
  ResReal_ptr u = m_u; \
  ResReal_ptr v = m_v; \
  ResReal_ptr w = m_w;


PLANCKIAN::PLANCKIAN(const RunParams& params)
  : KernelBase(rajaperf::Lcals_PLANCKIAN, params)
{
   setDefaultSize(100000);
   setDefaultReps(460);
}

PLANCKIAN::~PLANCKIAN() 
{
}

void PLANCKIAN::setUp(VariantID vid)
{
  allocAndInitData(m_x, getRunSize(), vid);
  allocAndInitData(m_y, getRunSize(), vid);
  allocAndInitData(m_u, getRunSize(), vid);
  allocAndInitData(m_v, getRunSize(), vid);
  allocAndInitDataConst(m_w, getRunSize(), 0.0, vid);
}

void PLANCKIAN::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      PLANCKIAN_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          PLANCKIAN_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      PLANCKIAN_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          PLANCKIAN_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      PLANCKIAN_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          PLANCKIAN_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      PLANCKIAN_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          PLANCKIAN_BODY;
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
      std::cout << "\n  PLANCKIAN : Unknown variant id = " << vid << std::endl;
    }

  }

}

void PLANCKIAN::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_w, getRunSize());
}

void PLANCKIAN::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_u);
  deallocData(m_v);
  deallocData(m_w);
}

} // end namespace lcals
} // end namespace rajaperf
