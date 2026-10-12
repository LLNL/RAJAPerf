//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DOT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

#include <iostream>

namespace rajaperf
{
namespace stream
{

template < size_t tune_idx >
void DOT::runOpenMPVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DOT_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DOT_1");
        Real_type dot = m_dot_init;

        #pragma omp parallel for reduction(+:dot)
        for (Index_type i = ibegin; i < iend; ++i ) {
          DOT_BODY;
        }

        m_dot += dot;
        RP_CALI_SUBKERNEL_END("DOT_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto dot_base_lam = [=](Index_type i) -> Real_type {
                            return a[i] * b[i];
                          };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DOT_1");
        Real_type dot = m_dot_init;

        #pragma omp parallel for reduction(+:dot)
        for (Index_type i = ibegin; i < iend; ++i ) {
          dot += dot_base_lam(i);
        }

        m_dot += dot;
        RP_CALI_SUBKERNEL_END("DOT_1");

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

          RP_CALI_SUBKERNEL_BEGIN("DOT_1");
          RAJA::ReduceSum<RAJA::omp_reduce, Real_type> dot(m_dot_init);

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
            DOT_BODY;
          });

          m_dot += dot;
          RP_CALI_SUBKERNEL_END("DOT_1");

        }
        stopTimer();

      } else if constexpr (tune_idx == 1) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("DOT_1");
          Real_type tdot = m_dot_init;

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::plus>(&tdot),
            [=] (Index_type i,
              RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& dot) {
              DOT_BODY;
            }
          );

          m_dot += static_cast<Real_type>(tdot);
          RP_CALI_SUBKERNEL_END("DOT_1");

        }
        stopTimer();

      } else {
        getCout() << "\n  DOT : Unknown OpenMP tuning index = " << tune_idx << std::endl;
      }

      break;
    }

    default : {
      getCout() << "\n  DOT : Unknown variant id = " << vid << std::endl;
    }

  }

}


void DOT::defineOpenMPVariantTunings()
{

  for (VariantID vid : {Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP}) {

    addVariantTuning<&DOT::runOpenMPVariant<0>>(
        vid, "default");

    if (vid == RAJA_OpenMP) {

      addVariantTuning<&DOT::runOpenMPVariant<1>>(
          vid, "new");

    }

  }
}

} // end namespace stream
} // end namespace rajaperf

#endif
