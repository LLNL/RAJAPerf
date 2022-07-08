//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MEMSET.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void MEMSET::runStdParVariantLibrary(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMSET_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::fill_n(std::execution::par_unseq,
                    x+ibegin, iend-ibegin, val);

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      camp::resources::Host res = camp::resources::Host::get_default();

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        res.memset(MEMSET_STD_ARGS);

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  MEMSET : Unknown variant id = " << vid << std::endl;
    }

  }
#endif
}

void MEMSET::runStdParVariantDefault(VariantID vid)
{
#if defined(RUN_STDPAR)
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMSET_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      auto begin = counting_iterator<Index_type>(ibegin);
      auto end   = counting_iterator<Index_type>(iend);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                       begin,end,
                       [=](Index_type i) {
          MEMSET_BODY;
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto memset_lambda = [=](Index_type i) {
                             MEMSET_BODY;
                           };

      auto begin = counting_iterator<Index_type>(ibegin);
      auto end   = counting_iterator<Index_type>(iend);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                       begin,end,
                       [=](Index_type i) {
          memset_lambda(i);
        });

      }
      stopTimer();

      break;
    }

#ifdef RAJA_ENABLE_STDPAR
    case RAJA_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::loop_exec>( RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
            MEMSET_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  MEMSET : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

void MEMSET::runStdParVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if (vid == Base_StdPar || vid == RAJA_StdPar) {

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

void MEMSET::setStdParTuningDefinitions(VariantID vid)
{
  if (vid == Base_StdPar || vid == RAJA_StdPar) {
    addVariantTuningName(vid, "library");
  }

  addVariantTuningName(vid, "default");
}

} // end namespace algorithm
} // end namespace rajaperf
