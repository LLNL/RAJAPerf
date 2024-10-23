//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void PRESSURE::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PRESSURE_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto pressure_lam1 = [=](Index_type i) {
                         PRESSURE_BODY1;
                       };
  auto pressure_lam2 = [=](Index_type i) {
                         PRESSURE_BODY2;
                       };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          PRESSURE_BODY1;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          PRESSURE_BODY2;
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       for (Index_type i = ibegin; i < iend; ++i ) {
         pressure_lam1(i);
       }

       for (Index_type i = ibegin; i < iend; ++i ) {
         pressure_lam2(i);
       }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::seq_region>( [=]() {

          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend), pressure_lam1);

          RAJA::forall<RAJA::seq_exec>( res,
            RAJA::RangeSegment(ibegin, iend), pressure_lam2);

        }); // end sequential region (for single-source code)

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  PRESSURE : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace apps
} // end namespace rajaperf
