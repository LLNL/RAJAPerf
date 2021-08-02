//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "COPY.hpp"

#include "RAJA/RAJA.hpp"

//#include <ranges>
#include <algorithm>
#include <execution>

#include <iostream>

namespace rajaperf 
{
namespace stream
{


void COPY::runStdParVariant(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  COPY_DATA_SETUP;

  //auto copy_lam = [=](Index_type i) {
  //                  COPY_BODY;
  //                };

  switch ( vid ) {

    case Base_StdPar : {

      //auto range = std::views::iota(ibegin, iend);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //std::for_each( std::execution::par_unseq,
        //                std::begin(range), std::end(range),
        //                [=](Index_type i) {
        //  COPY_BODY;
        //});

        std::copy( std::execution::par_unseq,
                   &a[ibegin], &a[iend], &c[ibegin]);
      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      //auto range = std::views::iota(ibegin, iend);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //std::for_each( std::execution::par_unseq,
        //                std::begin(range), std::end(range),
        //                [=](Index_type i) {
        //  copy_lam(i);
        //});
        std::copy( std::execution::par_unseq,
                   &a[ibegin], &a[iend], &c[ibegin]);
      }
      stopTimer();

      break;
    }

#ifdef RAJA_ENABLE_STDPAR
    case RAJA_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::stdpar_par_unseq_exec>(
          RAJA::RangeSegment(ibegin, iend), copy_lam);

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n  COPY : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace stream
} // end namespace rajaperf
