//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void INIT_VIEW1D::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INIT_VIEW1D_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          INIT_VIEW1D_BODY;
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto initview1d_base_lam = [=](Index_type i) {
                                   INIT_VIEW1D_BODY;
                                 };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          initview1d_base_lam(i);
        });
      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  INIT_VIEW1D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_STDPAR

