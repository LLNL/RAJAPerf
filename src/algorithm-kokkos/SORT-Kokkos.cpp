//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SORT.hpp"
#include <Kokkos_Sort.hpp>


#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void SORT::runKokkosVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SORT_DATA_SETUP;

  // Instantiate Kokkos Views

   auto x_view = getViewFromPointer(x, iend*run_reps);

#if defined (RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::sort(STD_SORT_ARGS);

      }
      stopTimer();

      break;
    }
/*
    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::sort<RAJA::loop_exec>(RAJA_SORT_ARGS);

      }
      stopTimer();

      break;
    }
*/

    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
		//#define STD_SORT_ARGS  vs. using RAJAPerf Suite expression
  		//x + iend*irep + ibegin, x + iend*irep + iend

		Kokkos::sort(x_view, iend*irep + ibegin, iend*irep + iend);
      }
      Kokkos::fence();
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  SORT : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

   moveDataToHostFromKokkosView(x, x_view, iend*run_reps);


}

} // end namespace algorithm
} // end namespace rajaperf
