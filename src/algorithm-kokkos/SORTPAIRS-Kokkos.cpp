//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SORTPAIRS.hpp"

#include "RAJA/RAJA.hpp"

#include <algorithm>
#include <vector>
#include <utility>
#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void SORTPAIRS::runKokkosVariant(VariantID vid)
{
        // Here, we are returning for configure, build and running purposes,
        // because Kokkos does not yet have a "sort pairs" capability
        return;

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SORTPAIRS_DATA_SETUP;


#if defined (RUN_KOKKOS)
  switch ( vid ) {

/*
    case Kokkos_Lambda : {
      
      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //RAJA::sort_pairs<RAJA::loop_exec>(RAJA_SORTPAIRS_ARGS);


                        });
      }
      Kokkos::fence();
      stopTimer();

      break;
    }
*/
    default : {
      std::cout << "\n  SORTPAIRS : Unknown variant id = " << vid << std::endl;
    }

  }
#endif // RUN_KOKKOS

  //moveDataToHostFromKokkosView(x, x_view, iend);
  //moveDataToHostFromKokkosView(i, i_view, iend);

}

} // end namespace algorithm
} // end namespace rajaperf
