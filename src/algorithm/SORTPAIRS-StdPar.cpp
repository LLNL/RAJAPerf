//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
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

  auto begin = counting_iterator<Index_type>(ibegin);
  auto end   = counting_iterator<Index_type>(iend);

  SORTPAIRS_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        using pair_type = std::pair<Real_type, Real_type>;

        std::vector<pair_type> vector_of_pairs;
        vector_of_pairs.reserve(iend-ibegin);

        for (Index_type iemp = ibegin; iemp < iend; ++iemp) {
          vector_of_pairs.emplace_back(x[iend*irep + iemp], i[iend*irep + iemp]);
        }

        std::sort( std::execution::par_unseq,
                   vector_of_pairs.begin(), vector_of_pairs.end(),
            [](pair_type const& lhs, pair_type const& rhs) {
              return lhs.first < rhs.first;
            });

        //for (Index_type iemp = ibegin; iemp < iend; ++iemp) {
        std::for_each( //std::execution::par_unseq,
                       begin, end,
                       [=](Index_type iemp) {
          const pair_type& pair = vector_of_pairs[iemp - ibegin];
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
