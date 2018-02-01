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

#include "EOS.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


#define EOS_DATA_SETUP_CPU \
  ResReal_ptr x = m_x; \
  ResReal_ptr y = m_y; \
  ResReal_ptr z = m_z; \
  ResReal_ptr u = m_u; \
\
  const Real_type q = m_q; \
  const Real_type r = m_r; \
  const Real_type t = m_t;


EOS::EOS(const RunParams& params)
  : KernelBase(rajaperf::Lcals_EOS, params)
{
   setDefaultSize(100000);
   setDefaultReps(5000);
}

EOS::~EOS() 
{
}

void EOS::setUp(VariantID vid)
{
  m_array_length = getRunSize() + 7;

  allocAndInitDataConst(m_x, m_array_length, 0.0, vid);
  allocAndInitData(m_y, m_array_length, vid);
  allocAndInitData(m_z, m_array_length, vid);
  allocAndInitData(m_u, m_array_length, vid);

  initData(m_q, vid);
  initData(m_r, vid);
  initData(m_t, vid);
}

void EOS::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      EOS_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          EOS_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      EOS_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          EOS_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      EOS_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          EOS_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      EOS_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          EOS_BODY;
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
      std::cout << "\n  EOS : Unknown variant id = " << vid << std::endl;
    }

  }

}

void EOS::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x, getRunSize());
}

void EOS::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_z);
  deallocData(m_u);
}

} // end namespace lcals
} // end namespace rajaperf
