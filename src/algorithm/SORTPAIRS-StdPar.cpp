//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SORTPAIRS.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <vector>
#include <utility>
#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void SORTPAIRS::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SORTPAIRS_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        using pair_type = std::pair<Real_type, Real_type>;

        std::vector<pair_type> vector_of_pairs;

#if 0
        vector_of_pairs.reserve(iend-ibegin);

        std::for_each_n( //std::execution::par, // parallelism leads to incorrectness
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=,&vector_of_pairs](Index_type iemp) noexcept {
          vector_of_pairs.emplace_back(x[iend*irep + iemp], i[iend*irep + iemp]);
        });
#else
        vector_of_pairs.resize(iend-ibegin);

        auto p = vector_of_pairs.data();
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type iemp) noexcept {
          p[iemp] = std::make_pair(x[iend*irep + iemp], i[iend*irep + iemp]);
        });
#endif

        std::sort( std::execution::par_unseq,
                   vector_of_pairs.begin(), vector_of_pairs.end(),
                   [](pair_type const& lhs, pair_type const& rhs) noexcept {
                     return lhs.first < rhs.first;
                   });

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(ibegin), iend-ibegin,
                         [=](Index_type iemp) noexcept {
          //const pair_type &pair = vector_of_pairs[iemp - ibegin];
          const pair_type &pair = p[iemp - ibegin];
          x[iend*irep + iemp] = pair.first;
          i[iend*irep + iemp] = pair.second;
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  SORTPAIRS : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace algorithm
} // end namespace rajaperf
