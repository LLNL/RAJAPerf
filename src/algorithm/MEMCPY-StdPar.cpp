//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MEMCPY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(BUILD_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void MEMCPY::runStdParVariantLibrary(VariantID vid)
{
#if defined(RUN_STDPAR)
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMCPY_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::copy_n(std::execution::par_unseq,
                    x+ibegin, iend-ibegin, y+ibegin);

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  MEMCPY : Unknown variant id = " << vid << std::endl;
    }

  }
#endif
}

void MEMCPY::runStdParVariantDefault(VariantID vid)
{
#if defined(RUN_STDPAR)
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMCPY_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          MEMCPY_BODY;
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto memcpy_lambda = [=](Index_type i) {
                             MEMCPY_BODY;
                           };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type i) {
          memcpy_lambda(i);
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  MEMCPY : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

void MEMCPY::runStdParVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if (vid == Base_StdPar) {

    if (tune_idx == t) {

      runStdParVariantLibrary(vid);

    }

    t += 1;

  }

  if (tune_idx == t) {

    runStdParVariantDefault(vid);

  }

  t += 1;
}

void MEMCPY::setStdParTuningDefinitions(VariantID vid)
{
  if (vid == Base_StdPar) {
    addVariantTuningName(vid, "library");
  }

  addVariantTuningName(vid, "default");
}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // BUILD_STDPAR

