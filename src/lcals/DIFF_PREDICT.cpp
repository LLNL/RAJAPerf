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

#include "DIFF_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


#define DIFF_PREDICT_DATA_SETUP_CPU \
  ResReal_ptr px = m_px; \
  ResReal_ptr cx = m_cx; \
  const Index_type offset = m_offset;


DIFF_PREDICT::DIFF_PREDICT(const RunParams& params)
  : KernelBase(rajaperf::Lcals_DIFF_PREDICT, params)
{
   setDefaultSize(100000);
   setDefaultReps(2000);
}

DIFF_PREDICT::~DIFF_PREDICT() 
{
}

void DIFF_PREDICT::setUp(VariantID vid)
{
  m_array_length = getRunSize() * 14;
  m_offset = getRunSize();

  allocAndInitDataConst(m_px, m_array_length, 0.0, vid);
  allocAndInitData(m_cx, m_array_length, vid);
}

void DIFF_PREDICT::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      DIFF_PREDICT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          DIFF_PREDICT_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      DIFF_PREDICT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          DIFF_PREDICT_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      DIFF_PREDICT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          DIFF_PREDICT_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      DIFF_PREDICT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          DIFF_PREDICT_BODY;
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
      std::cout << "\n  DIFF_PREDICT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void DIFF_PREDICT::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_px, m_array_length);
}

void DIFF_PREDICT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_px);
  deallocData(m_cx);
}

} // end namespace lcals
} // end namespace rajaperf
