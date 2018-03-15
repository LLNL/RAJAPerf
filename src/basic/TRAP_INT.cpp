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

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

//
// Function used in TRAP_INT loop.
//
RAJA_INLINE
Real_type trap_int_func(Real_type x,
                        Real_type y,
                        Real_type xp,
                        Real_type yp)
{
   Real_type denom = (x - xp)*(x - xp) + (y - yp)*(y - yp);
   denom = 1.0/sqrt(denom);
   return denom;
}


#define TRAP_INT_DATA_SETUP_CPU \
  Real_type x0 = m_x0; \
  Real_type xp = m_xp; \
  Real_type y = m_y; \
  Real_type yp = m_yp; \
  Real_type h = m_h;


TRAP_INT::TRAP_INT(const RunParams& params)
  : KernelBase(rajaperf::Basic_TRAP_INT, params)
{
   setDefaultSize(100000);
   setDefaultReps(2000);
}

TRAP_INT::~TRAP_INT() 
{
}

void TRAP_INT::setUp(VariantID vid)
{
  Real_type xn; 
  initData(xn, vid);

  initData(m_x0, vid);
  initData(m_xp, vid);
  initData(m_y,  vid);
  initData(m_yp, vid);

  m_h = xn - m_x0;

  m_sumx_init = 0.0;

  m_sumx = 0;
}

void TRAP_INT::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      TRAP_INT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type sumx = m_sumx_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          TRAP_INT_BODY;
        }

        m_sumx += sumx * h;

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      TRAP_INT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::seq_reduce, Real_type> sumx(m_sumx_init);

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](int i) {
          TRAP_INT_BODY;
        });

        m_sumx += static_cast<Real_type>(sumx.get()) * h;

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      TRAP_INT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type sumx = m_sumx_init;

        #pragma omp parallel for reduction(+:sumx)
        for (Index_type i = ibegin; i < iend; ++i ) {
          TRAP_INT_BODY;
        }

        m_sumx += sumx * h;

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      TRAP_INT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::omp_reduce, Real_type> sumx(m_sumx_init);

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          TRAP_INT_BODY;
        });

        m_sumx += static_cast<Real_type>(sumx.get()) * h;

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
      std::cout << "\n  TRAP_INT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void TRAP_INT::updateChecksum(VariantID vid)
{
  checksum[vid] += m_sumx;
}

void TRAP_INT::tearDown(VariantID vid)
{
  (void) vid;
}

} // end namespace basic
} // end namespace rajaperf
