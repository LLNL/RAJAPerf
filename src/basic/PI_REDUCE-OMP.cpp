//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t tune_idx >
void PI_REDUCE::runOpenMPVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_REDUCE_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("PI_REDUCE_1");
        Real_type pi = m_pi_init;

        #pragma omp parallel for reduction(+:pi)
        for (Index_type i = ibegin; i < iend; ++i ) {
          PI_REDUCE_BODY;
        }

        m_pi = 4.0 * pi;
        RP_CALI_SUBKERNEL_END("PI_REDUCE_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto pireduce_base_lam = [=](Index_type i, Real_type& pi) {
            PI_REDUCE_BODY;
          };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("PI_REDUCE_1");
        Real_type pi = m_pi_init;

        #pragma omp parallel for reduction(+:pi)
        for (Index_type i = ibegin; i < iend; ++i ) {
          pireduce_base_lam(i, pi);
        }

        m_pi = 4.0 * pi;
        RP_CALI_SUBKERNEL_END("PI_REDUCE_1");

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

          RP_CALI_SUBKERNEL_BEGIN("PI_REDUCE_1");
          RAJA::ReduceSum<RAJA::omp_reduce, Real_type> pi(m_pi_init);

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            [=](Index_type i) {
              PI_REDUCE_BODY;
          });

          m_pi = 4.0 * pi.get();
          RP_CALI_SUBKERNEL_END("PI_REDUCE_1");

        }
        stopTimer();

      } else if constexpr (tune_idx == 1) {

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("PI_REDUCE_1");
          Real_type tpi = m_pi_init;

          RAJA::forall<RAJA::omp_parallel_for_exec>( res,
            RAJA::RangeSegment(ibegin, iend),
            RAJA::expt::Reduce<RAJA::operators::plus>(&tpi),
            [=] (Index_type i,
              RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& pi) {
              PI_REDUCE_BODY;
            }
          );

          m_pi = static_cast<Real_type>(tpi) * 4.0;
          RP_CALI_SUBKERNEL_END("PI_REDUCE_1");

        }
        stopTimer();

      } else {
        getCout() << "\n  PI_REDUCE : Unknown OpenMP tuning index = " << tune_idx << std::endl;
      }

      break;
    }

    default : {
      getCout() << "\n  PI_REDUCE : Unknown variant id = " << vid << std::endl;
    }

  }

}


void PI_REDUCE::defineOpenMPVariantTunings()
{

  for (VariantID vid : {Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP}) {

    addVariantTuning<&PI_REDUCE::runOpenMPVariant<0>>(
        vid, "default");

    if (vid == RAJA_OpenMP) {

      addVariantTuning<&PI_REDUCE::runOpenMPVariant<1>>(
          vid, "new");

    }

  }
}

} // end namespace basic
} // end namespace rajaperf

#endif
