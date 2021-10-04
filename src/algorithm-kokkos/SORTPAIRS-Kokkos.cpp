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
        // FIXME
        return;

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SORTPAIRS_DATA_SETUP;

  // instatiate Kokkos Views
  // auto x_view = getViewFromPointer(x, iend);
   //auto i_view = getViewFromPointer(i, iend);

#if defined (RUN_KOKKOS)
  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        using pair_type = std::pair<Real_type, Real_type>;

        std::vector<pair_type> vector_of_pairs;
        vector_of_pairs.reserve(iend-ibegin);

        for (Index_type iemp = ibegin; iemp < iend; ++iemp) {
          vector_of_pairs.emplace_back(x[iend*irep + iemp], i[iend*irep + iemp]);
        }

        std::sort(vector_of_pairs.begin(), vector_of_pairs.end(),
            [](pair_type const& lhs, pair_type const& rhs) {
              return lhs.first < rhs.first;
            });

        for (Index_type iemp = ibegin; iemp < iend; ++iemp) {
          pair_type& pair = vector_of_pairs[iemp - ibegin];
          x[iend*irep + iemp] = pair.first;
          i[iend*irep + iemp] = pair.second;
        }

      }
      stopTimer();

      break;
    }
/*
    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::sort_pairs<RAJA::loop_exec>(RAJA_SORTPAIRS_ARGS);

      }
      stopTimer();

      break;
    }
    */
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
