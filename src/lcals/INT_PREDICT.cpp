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

#include "INT_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


#define INT_PREDICT_DATA_SETUP_CPU \
  ResReal_ptr px = m_px; \
  Real_type dm22 = m_dm22; \
  Real_type dm23 = m_dm23; \
  Real_type dm24 = m_dm24; \
  Real_type dm25 = m_dm25; \
  Real_type dm26 = m_dm26; \
  Real_type dm27 = m_dm27; \
  Real_type dm28 = m_dm28; \
  Real_type c0 = m_c0; \
  const Index_type offset = m_offset;


INT_PREDICT::INT_PREDICT(const RunParams& params)
  : KernelBase(rajaperf::Lcals_INT_PREDICT, params)
{
   setDefaultSize(100000);
   setDefaultReps(4000);
}

INT_PREDICT::~INT_PREDICT() 
{
}

void INT_PREDICT::setUp(VariantID vid)
{
  m_array_length = getRunSize() * 13;
  m_offset = getRunSize();

  m_px_initval = 1.0;
  allocAndInitDataConst(m_px, m_array_length, m_px_initval, vid);

  initData(m_dm22);
  initData(m_dm23);
  initData(m_dm24);
  initData(m_dm25);
  initData(m_dm26);
  initData(m_dm27);
  initData(m_dm28);
  initData(m_c0);
}

void INT_PREDICT::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      INT_PREDICT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          INT_PREDICT_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      INT_PREDICT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) { 
          INT_PREDICT_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      INT_PREDICT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          INT_PREDICT_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      INT_PREDICT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          INT_PREDICT_BODY;
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
      std::cout << "\n  INT_PREDICT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void INT_PREDICT::updateChecksum(VariantID vid)
{
  for (Index_type i = 0; i < getRunSize(); ++i) {
    m_px[i] -= m_px_initval;
  }
  
  checksum[vid] += calcChecksum(m_px, getRunSize());
}

void INT_PREDICT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_px);
}

} // end namespace lcals
} // end namespace rajaperf
