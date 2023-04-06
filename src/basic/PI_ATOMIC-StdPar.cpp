//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(BUILD_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{


void PI_ATOMIC::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_ATOMIC_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        *pi = m_pi_init;
        std::for_each_n( std::execution::par,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          rajaperf::Real_type x = (rajaperf::Real_type(i) + 0.5) * dx;
          ATOMIC_ADD(pi, dx / (1.0 + x * x));
        });
        *pi *= 4.0;

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto piatomic_base_lam = [=](Index_type i) {
                                 rajaperf::Real_type x = (rajaperf::Real_type(i) + 0.5) * dx;
                                 ATOMIC_ADD(pi, dx / (1.0 + x * x));
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        *pi = m_pi_init;
        std::for_each_n( std::execution::par,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                       [=](Index_type i) {
          piatomic_base_lam(i);
        });
        *pi *= 4.0;

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  PI_ATOMIC : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf

#endif  // BUILD_STDPAR

