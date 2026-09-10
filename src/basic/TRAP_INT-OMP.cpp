//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"

#include "TRAP_INT-func.hpp"

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t tune_idx >
void TRAP_INT::runOpenMPVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRAP_INT_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("TRAP_INT_1");
        Real_type sumx = m_sumx_init;

        #pragma omp parallel for reduction(+:sumx)
        for (Index_type i = ibegin; i < iend; ++i ) {
          TRAP_INT_BODY;
        }

        m_sumx += sumx * h;
        RP_CALI_SUBKERNEL_END("TRAP_INT_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto trapint_base_lam = [=](Index_type i) -> Real_type {
                                Real_type x = x0 + i*h;
                                return trap_int_func(x, y, xp, yp);
                              };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("TRAP_INT_1");
        Real_type sumx = m_sumx_init;

        #pragma omp parallel for reduction(+:sumx)
        for (Index_type i = ibegin; i < iend; ++i ) {
          sumx += trapint_base_lam(i);
        }

        m_sumx += sumx * h;
        RP_CALI_SUBKERNEL_END("TRAP_INT_1");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()}; 

      if constexpr (tune_idx == 0) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("TRAP_INT_1");
          RAJA::ReduceSum<RAJA::omp_reduce, Real_type> sumx(m_sumx_init);

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            TRAP_INT_BODY;
          });

          m_sumx += static_cast<Real_type>(sumx.get()) * h;
          RP_CALI_SUBKERNEL_END("TRAP_INT_1");

        }
        stopTimer();
  
      } else if constexpr (tune_idx == 1) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("TRAP_INT_1");
          Real_type tsumx = m_sumx_init;

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::plus>(&tsumx),
            [=] (Index_type i,
              RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& sumx) {
              TRAP_INT_BODY;
            }
          );

          m_sumx += static_cast<Real_type>(tsumx) * h;
          RP_CALI_SUBKERNEL_END("TRAP_INT_1");

        }
        stopTimer();

      } else {
        getCout() << "\n  TRAP_INT : Unknown OpenMP tuning index = " << tune_idx << std::endl;
      }

      break;
    }

    default : {
      getCout() << "\n  TRAP_INT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void TRAP_INT::defineOpenMPVariantTunings()
{

  for (VariantID vid : {Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP}) {

    addVariantTuning<&TRAP_INT::runOpenMPVariant<0>>(
        vid, "default");

    if (vid == RAJA_OpenMP) {

      addVariantTuning<&TRAP_INT::runOpenMPVariant<1>>(
          vid, "new");

    }

  }
}

} // end namespace basic
} // end namespace rajaperf

#endif
