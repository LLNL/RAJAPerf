//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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


void SORTPAIRS::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  SORTPAIRS_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("SORTPAIRS_1");
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
        RP_CALI_SUBKERNEL_END("SORTPAIRS_1");

      }
      stopTimer();

      break;
    }
#if defined(RUN_RAJA_SEQ)
    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("SORTPAIRS_1");
        RAJA::sort_pairs<RAJA::seq_exec>(res, RAJA_SORTPAIRS_ARGS);
        RP_CALI_SUBKERNEL_END("SORTPAIRS_1");

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  SORTPAIRS : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(SORTPAIRS, Seq, Base_Seq, RAJA_Seq)

} // end namespace algorithm
} // end namespace rajaperf
