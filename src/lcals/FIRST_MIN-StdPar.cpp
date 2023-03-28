//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_MIN.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void FIRST_MIN::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  FIRST_MIN_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if 0
        auto result =
        std::min_element( std::execution::par_unseq,
                          &x[ibegin], &x[iend]);
        auto loc = std::distance(&x[ibegin], result);
#else
        for (Index_type i = ibegin; i < iend; ++i ) {
          if ( x[i] < mymin.val ) {
            mymin.val = x[i];
            mymin.loc = i;
          }
        }
#endif

        m_minloc = std::max(m_minloc, loc);

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto firstmin_base_lam = [=](Index_type i) -> Real_type {
                                 return x[i];
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        FIRST_MIN_MINLOC_INIT;

        for (Index_type i = ibegin; i < iend; ++i ) {
          if ( firstmin_base_lam(i) < mymin.val ) {
            mymin.val = x[i];
            mymin.loc = i;
          }
        }

        m_minloc = std::max(m_minloc, mymin.loc);

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  FIRST_MIN : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace lcals
} // end namespace rajaperf
