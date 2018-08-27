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

#include "DAXPY.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{


#define DAXPY_DATA_SETUP_CPU \
  ResReal_ptr x = m_x; \
  ResReal_ptr y = m_y; \
  Real_type a = m_a; 


DAXPY::DAXPY(const RunParams& params)
  : KernelBase(rajaperf::Basic_DAXPY, params)
{
   setDefaultSize(100000);
   setDefaultReps(5000);
}

DAXPY::~DAXPY() 
{
}

void DAXPY::setUp(VariantID vid)
{
  allocAndInitDataConst(m_y, getRunSize(), 0.0, vid);
  allocAndInitData(m_x, getRunSize(), vid);
  initData(m_a);
}

void DAXPY::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      DAXPY_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          DAXPY_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      DAXPY_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          DAXPY_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      DAXPY_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          DAXPY_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      DAXPY_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          DAXPY_BODY;
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
      std::cout << "\n  DAXPY : Unknown variant id = " << vid << std::endl;
    }

  }

}

void DAXPY::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_y, getRunSize());
}

void DAXPY::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
}

} // end namespace basic
} // end namespace rajaperf
