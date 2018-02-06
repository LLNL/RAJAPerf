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

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

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

//
// Define thread block size for target execution
//
#define NUMTEAMS 128

#define TRAP_INT_DATA_SETUP_OMP_TARGET \
  Real_type x0 = m_x0; \
  Real_type xp = m_xp; \
  Real_type y = m_y; \
  Real_type yp = m_yp; \
  Real_type h = m_h;

#define TRAP_INT_DATA_TEARDOWN_OMP_TARGET // nothing to do here...


void TRAP_INT::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_OpenMPTarget ) {

    TRAP_INT_DATA_SETUP_OMP_TARGET;

    #pragma omp target data map(to:x0,xp,y,yp,h)

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type sumx = m_sumx_init;

      #pragma omp target teams distribute parallel for map(tofrom: sumx) reduction(+:sumx) \
                         num_teams(NUMTEAMS) schedule(static, 1) 
        
      for (Index_type i = ibegin; i < iend; ++i ) {
        TRAP_INT_BODY;
      }

      m_sumx += sumx * h;

    }
    stopTimer();

    #pragma omp target exit data map(delete: x0,xp,y,yp,h) 

  } else if ( vid == RAJA_OpenMPTarget ) {

    TRAP_INT_DATA_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::omp_target_reduce<NUMTEAMS>, Real_type> sumx(m_sumx_init);

      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        TRAP_INT_BODY;
      });

      m_sumx += static_cast<Real_type>(sumx.get()) * h;

    }
    stopTimer();

    TRAP_INT_DATA_TEARDOWN_OMP_TARGET;

  } else {
     std::cout << "\n  TRAP_INT : Unknown OMP Targetvariant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
